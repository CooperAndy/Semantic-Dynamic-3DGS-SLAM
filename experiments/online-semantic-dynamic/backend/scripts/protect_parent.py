"""Protect explicitly configured inputs from worker filesystem mutations."""
import os
import sys
from pathlib import Path

ROOTS = [
    Path(root).resolve()
    for root in os.environ.get("S23_PROTECTED_ROOTS", "").split(os.pathsep)
    if root
]


def forbidden(path):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    resolved = Path(os.fsdecode(path)).resolve()
    return any(resolved.is_relative_to(root) for root in ROOTS)


def install():
    def hook(event, args):
        if event == "open":
            mode = args[1] or ""
            flags = args[2] or 0
            write = any(flag in mode for flag in "wax+") or flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            )
            if write and forbidden(args[0]):
                raise RuntimeError("Write to a protected input path")
        elif event in {"os.rename", "os.remove", "os.rmdir", "os.mkdir", "os.link", "os.symlink"}:
            if any(forbidden(path) for path in args[:2]):
                raise RuntimeError("Mutation of a protected input path")
    sys.addaudithook(hook)
