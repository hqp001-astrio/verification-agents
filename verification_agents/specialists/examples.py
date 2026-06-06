"""Toy benchmark diffs covering the three verdict classes.

Used by the runner and tests. Each is a minimal unified diff so ``parse_diff`` can
discover the property and the specialist pipeline can verify it end to end.
"""

from __future__ import annotations

# NO: off-by-one — the loop drives the index to len(items), out of bounds.
BUG_BOUNDS = """\
diff --git a/app/items.py b/app/items.py
new file mode 100644
--- /dev/null
+++ b/app/items.py
@@ -0,0 +1,5 @@
+def get_item(items, index):
+    for index in range(len(items) + 1):
+        value = items[index]
+    return value
"""

# YES: the access is guarded by an explicit 0 <= index < len(items) check.
SAFE_BOUNDS = """\
diff --git a/app/items.py b/app/items.py
new file mode 100644
--- /dev/null
+++ b/app/items.py
@@ -0,0 +1,4 @@
+def get_item(items, index):
+    if 0 <= index < len(items):
+        return items[index]
+    return None
"""

# UNSURE: the sequence comes from an external service not visible in this diff, so
# Z3 may flag a possible violation but the counterexample cannot be grounded by
# execution (the dependency is out of scope) -> honestly inconclusive.
UNSURE_EXTERNAL = """\
diff --git a/app/remote.py b/app/remote.py
new file mode 100644
--- /dev/null
+++ b/app/remote.py
@@ -0,0 +1,3 @@
+def get_item(index):
+    items = load_remote_items()
+    return items[index]
"""

EXAMPLES = {
    "bug": BUG_BOUNDS,
    "safe": SAFE_BOUNDS,
    "unsure": UNSURE_EXTERNAL,
}
