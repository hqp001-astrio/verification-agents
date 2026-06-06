from __future__ import annotations

import re
import uuid

import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
from tree_sitter import Language as TSLanguage
from tree_sitter import Parser

from verification_agents.models import (
    CallEdge,
    CodeAnalysis,
    CodeUnit,
    Language,
    PropertyKind,
    VerifiableProperty,
)

_PY_LANG = TSLanguage(tspython.language())
_JS_LANG = TSLanguage(tsjavascript.language())


def _detect_language(filename: str) -> Language:
    if filename.endswith(".py"):
        return Language.PYTHON
    if filename.endswith((".js", ".ts", ".jsx", ".tsx")):
        return Language.JAVASCRIPT
    return Language.UNKNOWN


def _parse_diff_headers(diff: str) -> dict[str, list[tuple[int, int]]]:
    """Return {filename: [(start_line, end_line), ...]} for added/changed hunks."""
    result: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    for line in diff.splitlines():
        m = re.match(r"^\+\+\+ b/(.+)$", line)
        if m:
            current_file = m.group(1)
            result.setdefault(current_file, [])
            continue
        m = re.match(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", line)
        if m and current_file:
            start = int(m.group(1))
            count = int(m.group(2)) if m.group(2) is not None else 1
            result[current_file].append((start, start + count - 1))
    return result


def _extract_full_source_from_diff(diff: str, filename: str) -> str:
    """Reconstruct the new version of a file from its diff hunks."""
    lines: list[str] = []
    in_file = False
    for line in diff.splitlines():
        if re.match(r"^\+\+\+ b/" + re.escape(filename) + r"$", line):
            in_file = True
            continue
        if in_file and line.startswith("+++ "):
            break
        if in_file:
            if line.startswith("diff ") or (line.startswith("+++ ") and filename not in line):
                break
            if line.startswith("@@") or line.startswith("---"):
                continue
            if line.startswith("+"):
                lines.append(line[1:])
            elif line.startswith("-"):
                continue
            else:
                lines.append(line[1:] if line.startswith(" ") else line)
    return "\n".join(lines)


def _get_ts_parser(lang: Language) -> Parser | None:
    if lang == Language.PYTHON:
        p = Parser(_PY_LANG)
        return p
    if lang == Language.JAVASCRIPT:
        p = Parser(_JS_LANG)
        return p
    return None


def _overlaps(node_start: int, node_end: int, hunks: list[tuple[int, int]]) -> bool:
    for h_start, h_end in hunks:
        if node_start <= h_end and node_end >= h_start:
            return True
    return False


def _extract_units_and_edges(
    source: str,
    filename: str,
    lang: Language,
    changed_hunks: list[tuple[int, int]],
) -> tuple[list[CodeUnit], list[CallEdge]]:
    parser = _get_ts_parser(lang)
    if parser is None:
        return [], []

    tree = parser.parse(source.encode())
    source_lines = source.splitlines()
    units: list[CodeUnit] = []
    edges: list[CallEdge] = []

    func_types = {
        Language.PYTHON: {"function_definition"},
        Language.JAVASCRIPT: {"function_declaration", "function_expression",
                              "arrow_function", "method_definition"},
    }.get(lang, set())

    call_types = {
        Language.PYTHON: {"call"},
        Language.JAVASCRIPT: {"call_expression"},
    }.get(lang, set())

    def node_name(node) -> str:
        for child in node.children:
            if child.type == "identifier":
                return child.text.decode()
        return "<anonymous>"

    def walk(node, unit_name: str | None = None):
        if node.type in func_types:
            start = node.start_point[0] + 1
            end = node.end_point[0] + 1
            name = node_name(node)
            if _overlaps(start, end, changed_hunks):
                unit_source = "\n".join(source_lines[start - 1 : end])
                units.append(CodeUnit(
                    name=name,
                    filename=filename,
                    language=lang,
                    start_line=start,
                    end_line=end,
                    source=unit_source,
                ))
            for child in node.children:
                walk(child, name)
            return

        if node.type in call_types and unit_name:
            callee = ""
            if node.children:
                fn = node.children[0]
                if fn.type == "identifier":
                    callee = fn.text.decode()
                elif fn.type == "attribute":
                    callee = fn.text.decode()
            if callee:
                edges.append(CallEdge(caller=unit_name, callee=callee, filename=filename))

        for child in node.children:
            walk(child, unit_name)

    walk(tree.root_node)
    return units, edges


_ARRAY_SUBSCRIPT = {
    Language.PYTHON: "subscript",
    Language.JAVASCRIPT: "subscript_expression",
}
_ATTR_ACCESS = {
    Language.PYTHON: "attribute",
    Language.JAVASCRIPT: "member_expression",
}
_ARITH_OPS = {"+", "-", "*", "/", "%", "**", "<<", ">>"}
_LOOP_TYPES = {
    Language.PYTHON: {"for_statement", "while_statement"},
    Language.JAVASCRIPT: {"for_statement", "while_statement", "for_in_statement",
                          "for_of_statement"},
}


def _discover_properties(
    unit: CodeUnit,
    lang: Language,
) -> list[VerifiableProperty]:
    parser = _get_ts_parser(lang)
    if parser is None:
        return []

    tree = parser.parse(unit.source.encode())
    props: list[VerifiableProperty] = []

    def walk(node):
        t = node.type
        if t == _ARRAY_SUBSCRIPT.get(lang):
            props.append(VerifiableProperty(
                id=str(uuid.uuid4()),
                kind=PropertyKind.ARRAY_BOUNDS,
                unit_name=unit.name,
                filename=unit.filename,
                start_line=unit.start_line + node.start_point[0],
                description=f"Array/sequence index access in `{unit.name}`: "
                            f"`{node.text.decode()[:80]}` — verify index is within bounds",
            ))
        elif t == _ATTR_ACCESS.get(lang):
            props.append(VerifiableProperty(
                id=str(uuid.uuid4()),
                kind=PropertyKind.NULL_DEREFERENCE,
                unit_name=unit.name,
                filename=unit.filename,
                start_line=unit.start_line + node.start_point[0],
                description=f"Attribute/member access in `{unit.name}`: "
                            f"`{node.text.decode()[:80]}` — verify object is not None/null",
            ))
        elif t == "binary_operator" and lang == Language.PYTHON:
            # the operator child's node type is the symbol itself (e.g. "/"), not
            # "operator", so fetch it by field name.
            op_node = node.child_by_field_name("operator")
            op = op_node.text.decode() if op_node else ""
            if op in {"/", "//", "%"}:
                props.append(VerifiableProperty(
                    id=str(uuid.uuid4()),
                    kind=PropertyKind.DIVISION_BY_ZERO,
                    unit_name=unit.name,
                    filename=unit.filename,
                    start_line=unit.start_line + node.start_point[0],
                    description=f"Division/modulo in `{unit.name}`: "
                                f"`{node.text.decode()[:80]}` — verify divisor is non-zero",
                ))
            elif op in _ARITH_OPS:
                props.append(VerifiableProperty(
                    id=str(uuid.uuid4()),
                    kind=PropertyKind.INTEGER_OVERFLOW,
                    unit_name=unit.name,
                    filename=unit.filename,
                    start_line=unit.start_line + node.start_point[0],
                    description=f"Arithmetic operation in `{unit.name}`: "
                                f"`{node.text.decode()[:80]}` — verify no integer overflow",
                ))
        elif t in _LOOP_TYPES.get(lang, set()):
            props.append(VerifiableProperty(
                id=str(uuid.uuid4()),
                kind=PropertyKind.LOOP_TERMINATION,
                unit_name=unit.name,
                filename=unit.filename,
                start_line=unit.start_line + node.start_point[0],
                description=f"Loop in `{unit.name}` — verify loop terminates",
            ))
        for child in node.children:
            walk(child)

    walk(tree.root_node)

    # Deduplicate by kind within the same unit (keep first occurrence per kind)
    seen: set[tuple[str, str]] = set()
    deduped: list[VerifiableProperty] = []
    for p in props:
        key = (p.unit_name, p.kind)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    return deduped


def run(diff: str) -> CodeAnalysis:
    file_hunks = _parse_diff_headers(diff)
    all_units: list[CodeUnit] = []
    all_edges: list[CallEdge] = []
    all_props: list[VerifiableProperty] = []

    for filename, hunks in file_hunks.items():
        lang = _detect_language(filename)
        if lang == Language.UNKNOWN:
            continue
        source = _extract_full_source_from_diff(diff, filename)
        if not source.strip():
            continue
        units, edges = _extract_units_and_edges(source, filename, lang, hunks)
        all_units.extend(units)
        all_edges.extend(edges)
        for unit in units:
            all_props.extend(_discover_properties(unit, lang))

    return CodeAnalysis(units=all_units, call_edges=all_edges, properties=all_props)
