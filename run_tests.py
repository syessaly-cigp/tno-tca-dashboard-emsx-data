"""Offline test runner: executes tests/test_pipeline.py without a pytest install.

This environment has no network access to `pip install pytest`. This shim provides
just enough of the pytest API (approx, mark.skipif, raises) to run the suite. When
pytest is available, prefer `python -m pytest -q`.
"""
from __future__ import annotations

import sys
import types
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ---- minimal pytest stub -----------------------------------------------------
class _Approx:
    def __init__(self, expected, rel=1e-6, abs=1e-12):
        self.expected, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        tol = max(self.abs, self.rel * abs(self.expected))
        return abs(other - self.expected) <= tol

    def __repr__(self):
        return f"approx({self.expected}, rel={self.rel})"


class _Skip(Exception):
    pass


class _Mark:
    @staticmethod
    def skipif(condition, reason=""):
        def deco(fn):
            fn.__skip__ = (bool(condition), reason)
            return fn
        return deco


pytest_stub = types.ModuleType("pytest")
pytest_stub.approx = lambda expected, rel=1e-6, abs=1e-12: _Approx(expected, rel, abs)
pytest_stub.mark = _Mark()


class _Raises:
    def __init__(self, exc):
        self.exc = exc

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        return et is not None and issubclass(et, self.exc)


pytest_stub.raises = lambda exc: _Raises(exc)
sys.modules["pytest"] = pytest_stub


def main() -> int:
    import importlib.util

    spec = importlib.util.spec_from_file_location("test_pipeline", ROOT / "tests" / "test_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    tests = [getattr(mod, n) for n in dir(mod) if n.startswith("test_")]
    passed = skipped = failed = 0
    for fn in tests:
        cond, reason = getattr(fn, "__skip__", (False, ""))
        if cond:
            print(f"SKIP {fn.__name__}: {reason}")
            skipped += 1
            continue
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
