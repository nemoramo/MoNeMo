#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class Finding:
    path: Path
    line: int
    column: int
    codepoint: int

    def format(self) -> str:
        name = unicodedata.name(chr(self.codepoint), "<UNKNOWN>")
        return f"{self.path}:{self.line}:{self.column}: U+{self.codepoint:04X} {name}"


def iter_tracked_files() -> List[Path]:
    out = subprocess.check_output(["git", "ls-files"], text=True)
    return [Path(p) for p in out.splitlines() if p.strip()]


def looks_binary(data: bytes) -> bool:
    return b"\x00" in data


def find_unicode_controls(text: str, path: Path) -> List[Finding]:
    findings: List[Finding] = []
    for line_no, line in enumerate(text.splitlines(True), start=1):
        for col_no, ch in enumerate(line, start=1):
            if unicodedata.category(ch) == "Cf":
                findings.append(Finding(path=path, line=line_no, column=col_no, codepoint=ord(ch)))
    return findings


def strip_unicode_controls(text: str) -> str:
    return "".join(ch for ch in text if unicodedata.category(ch) != "Cf")


def process_paths(paths: Sequence[Path], fix: bool) -> List[Finding]:
    all_findings: List[Finding] = []
    for path in paths:
        if not path.exists() or path.is_dir():
            continue

        try:
            data = path.read_bytes()
        except OSError:
            continue

        if looks_binary(data):
            continue

        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue

        findings = find_unicode_controls(text, path)
        if not findings:
            continue

        all_findings.extend(findings)

        if fix:
            fixed = strip_unicode_controls(text)
            if fixed != text:
                path.write_text(fixed, encoding="utf-8")

    return all_findings


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect and optionally remove Unicode control (Cf) characters.")
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Remove Unicode control characters in-place for the provided files.",
    )
    parser.add_argument(
        "--all-files",
        action="store_true",
        help="Scan all git-tracked files (ignores positional file arguments).",
    )
    parser.add_argument("files", nargs="*", help="Files to scan (as passed by pre-commit).")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if args.all_files:
        paths = iter_tracked_files()
    else:
        paths = [Path(p) for p in args.files]

    findings = process_paths(paths, fix=args.fix)
    if not findings:
        return 0

    for finding in findings:
        print(finding.format())

    # Return non-zero if any findings exist (even after --fix) so callers notice and re-stage files.
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

