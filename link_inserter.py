"""
link_inserter.py
----------------
Step 2 of the PDF pipeline.

Reads a _scan.json produced by pdf_scan.py, finds vault notes whose frontmatter
contains a matching 'source:' field, and appends Obsidian page links to caption
label lines.

Usage:
    python link_inserter.py <scan_json> <vault_directory>

Example:
    python link_inserter.py "C:\\vault\\ORD-5700_scan.json" "C:\\vault\\Inbox"

What it changes:
    Caption lines are the only lines modified. A caption line is one that starts
    with a label from the JSON followed by a colon, e.g.:
        Figure 3-2: Limits for extrusion
    becomes:
        Figure 3-2: Limits for extrusion [[ORD-5700.pdf#page=47|↗]]

What it does NOT change:
    - Inline mentions (Figure 3-2 in body text) — left as plain text.
    - Lines already containing ↗ — skipped so re-runs are safe.
    - Labels stored as lists in the JSON (ambiguous, need manual resolution).
    - The source PDF or any other file.
"""

import re
import sys
import json
from pathlib import Path

# Normalise typographic dashes to plain hyphens for label matching.
# Handles the case where the PDF used en-dashes but the note uses hyphens.
_DASH_MAP = str.maketrans('\u2013\u2014', '--')


def _norm(s: str) -> str:
    return s.translate(_DASH_MAP).casefold()


def read_source_field(md_path: Path) -> str | None:
    """Return the value of 'source:' from YAML frontmatter, or None."""
    try:
        text = md_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return None
    if not text.startswith('---'):
        return None
    close = text.find('\n---', 3)
    if close == -1:
        return None
    for line in text[3:close].splitlines():
        m = re.match(r"^\s*source\s*:\s*['\"]?(.+?)['\"]?\s*$", line)
        if m:
            return m.group(1).strip()
    return None


def process_note(md_path: Path, label_to_page: dict, pdf_name: str) -> tuple[int, list[str]]:
    """
    Append page links to caption label lines in md_path.
    Returns (count_inserted, warnings).
    Already-linked lines (containing ↗) are not touched.
    """
    try:
        text = md_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        return 0, [f"Cannot read {md_path.name}: {exc}"]

    lines = text.splitlines(keepends=True)
    count = 0

    for idx, line in enumerate(lines):
        bare = line.rstrip('\n\r')
        if '\u2197' in bare:    # ↗ already present — skip
            continue

        norm_bare = _norm(bare)

        for label, page in label_to_page.items():
            # Match lines that start with this label then optional space and colon.
            # Normalised comparison handles dash-style mismatches between PDF and note.
            if re.match(r'^' + re.escape(_norm(label)) + r'\s*:', norm_bare):
                link = f' [[{pdf_name}#page={page}|\u2197]]'
                ending = line[len(bare):]           # preserve original line ending
                lines[idx] = bare + link + ending
                count += 1
                break   # at most one label can match a given line

    if count:
        try:
            md_path.write_text(''.join(lines), encoding='utf-8')
        except OSError as exc:
            return 0, [f"Cannot write {md_path.name}: {exc}"]

    return count, []


def main():
    if len(sys.argv) < 3:
        print('Usage: python link_inserter.py <scan_json> <vault_directory>')
        print('Example: python link_inserter.py "C:\\vault\\ORD-5700_scan.json" "C:\\vault\\Inbox"')
        sys.exit(1)

    json_path = Path(sys.argv[1])
    vault_dir = Path(sys.argv[2])

    if not json_path.exists():
        print(f'ERROR: File not found: {json_path}')
        sys.exit(1)
    if not vault_dir.is_dir():
        print(f'ERROR: Directory not found: {vault_dir}')
        sys.exit(1)

    stem = json_path.stem
    if not stem.endswith('_scan'):
        print(f'ERROR: Expected a file named <name>_scan.json, got: {json_path.name}')
        sys.exit(1)
    pdf_name = stem[:-5] + '.pdf'   # "ORD-5700_scan" → "ORD-5700.pdf"

    try:
        with open(json_path, encoding='utf-8') as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f'ERROR: Cannot read scan JSON: {exc}')
        sys.exit(1)

    label_to_page: dict[str, int] = {}
    ambiguous: list[str] = []
    for label, value in raw.items():
        if isinstance(value, int):
            label_to_page[label] = value
        else:
            ambiguous.append(label)

    print(f'PDF:       {pdf_name}')
    print(f'Labels:    {len(label_to_page)} ready  |  {len(ambiguous)} ambiguous (skipped)')
    if ambiguous:
        print(f'           Ambiguous labels must be resolved in {json_path.name} before they can be linked.')
    print(f'Vault:     {vault_dir}')
    print()

    matching: list[Path] = [
        md for md in vault_dir.rglob('*.md')
        if (read_source_field(md) or '').lower() == pdf_name.lower()
    ]

    if not matching:
        print(f'No notes found with  source: "{pdf_name}"  in {vault_dir}')
        print('Check that the source field in your note frontmatter matches the PDF filename exactly.')
        sys.exit(0)

    print(f'Notes matched: {len(matching)}')
    print()

    total = 0
    warnings: list[str] = []

    for note in sorted(matching):
        n, w = process_note(note, label_to_page, pdf_name)
        warnings.extend(w)
        tag = f'{n} link(s) inserted' if n else 'no new caption lines found'
        print(f'  {note.name:<55} {tag}')
        total += n

    if warnings:
        print()
        print('WARNINGS:')
        for w in warnings:
            print(f'  {w}')

    print()
    print(f'Done.  {total} link(s) inserted across {len(matching)} note(s).')

    if ambiguous:
        print()
        print(f'Skipped {len(ambiguous)} ambiguous label(s).  To process them:')
        print(f'  1. Open {json_path.name}')
        print(f'  2. For each list value, replace it with the correct single page number.')
        print(f'  3. Re-run this script — lines already containing ↗ will not be touched.')


if __name__ == '__main__':
    main()
