"""
pdf_scan.py
-----------
Scans a PDF for figure, table, and other labeled elements.
Outputs a JSON lookup file and a human-readable log for review.

Usage:
    python pdf_scan.py "C:\path\to\your\file.pdf"

Outputs (written to same folder as the PDF):
    file_scan.json   — machine-readable label → page number lookup
    file_scan_log.txt — human-readable summary for review before running link_inserter.py

No changes are made to any vault files. This script is read-only.
"""

import re
import sys
import json
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber is not installed.")
    print("Run:  pip install pdfplumber")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Label patterns — add or remove patterns here to tune for different documents
# Each entry is (pattern_name, compiled_regex)
# The regex must have a named group called 'label' capturing the full label text
# ---------------------------------------------------------------------------
LABEL_PATTERNS = [
    # "Figure 3-3:", "Figure 3-3 —", "Figure 3-3 " (space before title)
    ("Figure",  re.compile(r'(?i)\b(fig(?:ure)?\.?\s+\d+[\-–]\d+(?:\.\d+)*(?:\s*[:\-–]|\s))', re.MULTILINE)),
    # "Figure 3:" or "Figure 3 " (single number, no section prefix)
    ("Figure",  re.compile(r'(?i)\b(fig(?:ure)?\.?\s+\d+(?:\s*[:\-–]|\s))', re.MULTILINE)),
    # "Table 4-2:", "Table 4-2 —"
    ("Table",   re.compile(r'(?i)\b(table\s+\d+[\-–]\d+(?:\.\d+)*(?:\s*[:\-–]|\s))', re.MULTILINE)),
    # "Table 4:" or "Table 4 "
    ("Table",   re.compile(r'(?i)\b(table\s+\d+(?:\s*[:\-–]|\s))', re.MULTILINE)),
    # "Chart 2-1:", "Diagram 5-3:"
    ("Chart",   re.compile(r'(?i)\b(chart\s+\d+[\-–]?\d*(?:\s*[:\-–]|\s))', re.MULTILINE)),
    ("Diagram", re.compile(r'(?i)\b(diagram\s+\d+[\-–]?\d*(?:\s*[:\-–]|\s))', re.MULTILINE)),
    # "Drawing 1042:", "Drawing No. 3"
    ("Drawing", re.compile(r'(?i)\b(drawing(?:\s+no\.?)?\s+[\w\-]+(?:\s*[:\-–]|\s))', re.MULTILINE)),
]

# Minimum characters a page must yield to be considered text-bearing
# Pages below this are likely image-only / scanned and will be flagged
MIN_TEXT_LENGTH = 50


def normalize_label(raw: str) -> str:
    """Strip trailing punctuation/whitespace, collapse internal spaces."""
    label = raw.strip().rstrip(':–—- ')
    label = re.sub(r'\s+', ' ', label)
    return label


def scan_pdf(pdf_path: Path) -> dict:
    """
    Scan pdf_path page by page.
    Returns a dict with keys:
        'found'   — {normalized_label: [page_numbers]}
        'skipped' — [page_numbers that had insufficient text]
        'stats'   — summary counts
    """
    found = {}       # label → list of page numbers where it appears
    skipped = []     # page numbers with too little text (likely images)
    total_pages = 0

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            page_num = i + 1  # 1-indexed
            text = page.extract_text() or ""

            if len(text.strip()) < MIN_TEXT_LENGTH:
                skipped.append(page_num)
                continue

            for _pattern_name, pattern in LABEL_PATTERNS:
                for match in pattern.finditer(text):
                    raw = match.group(1)
                    label = normalize_label(raw)
                    if not label:
                        continue
                    if label not in found:
                        found[label] = []
                    if page_num not in found[label]:
                        found[label].append(page_num)

    # Flag labels that appear on more than one page (cross-references vs. actual captions)
    duplicates = {k: v for k, v in found.items() if len(v) > 1}

    return {
        "found": found,
        "skipped": skipped,
        "duplicates": duplicates,
        "stats": {
            "total_pages": total_pages,
            "pages_skipped_no_text": len(skipped),
            "unique_labels_found": len(found),
            "labels_on_multiple_pages": len(duplicates),
        }
    }


def write_outputs(pdf_path: Path, results: dict) -> tuple[Path, Path]:
    """Write JSON lookup and human-readable log. Returns (json_path, log_path)."""
    stem = pdf_path.stem
    out_dir = pdf_path.parent

    json_path = out_dir / f"{stem}_scan.json"
    log_path  = out_dir / f"{stem}_scan_log.txt"

    # --- JSON output ---
    # Format: {label: first_page} for unambiguous labels
    #         {label: [page1, page2, ...]} for duplicates (needs manual resolution)
    json_out = {}
    for label, pages in results["found"].items():
        if len(pages) == 1:
            json_out[label] = pages[0]
        else:
            json_out[label] = pages  # list signals ambiguity

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)

    # --- Human-readable log ---
    stats = results["stats"]
    lines = []
    lines.append("=" * 70)
    lines.append(f"PDF SCAN REPORT")
    lines.append(f"File: {pdf_path.name}")
    lines.append("=" * 70)
    lines.append(f"Total pages:              {stats['total_pages']}")
    lines.append(f"Pages with no text:       {stats['pages_skipped_no_text']}")
    lines.append(f"Unique labels found:      {stats['unique_labels_found']}")
    lines.append(f"Labels on multiple pages: {stats['labels_on_multiple_pages']}")
    lines.append("")

    if results["skipped"]:
        lines.append("─" * 70)
        lines.append("PAGES WITH NO EXTRACTABLE TEXT (likely images or scanned):")
        lines.append("  These pages were not scanned for labels.")
        # Print as ranges for readability
        pages = sorted(results["skipped"])
        ranges = []
        start = end = pages[0]
        for p in pages[1:]:
            if p == end + 1:
                end = p
            else:
                ranges.append(f"{start}" if start == end else f"{start}-{end}")
                start = end = p
        ranges.append(f"{start}" if start == end else f"{start}-{end}")
        lines.append(f"  Pages: {', '.join(ranges)}")
        lines.append("")

    if results["duplicates"]:
        lines.append("─" * 70)
        lines.append("LABELS FOUND ON MULTIPLE PAGES (ACTION REQUIRED):")
        lines.append("  These appear more than once — likely cross-referenced in body text.")
        lines.append("  In the JSON file these are stored as lists, not single page numbers.")
        lines.append("  Review each one and manually edit the JSON to keep only the caption page.")
        lines.append("")
        for label, pages in sorted(results["duplicates"].items()):
            lines.append(f"  {label:<35} pages: {pages}")
        lines.append("")

    lines.append("─" * 70)
    lines.append("ALL LABELS FOUND (label → page number):")
    lines.append("  Review this list before running link_inserter.py.")
    lines.append("  If a label looks wrong (garbled text, false match), delete it from the JSON.")
    lines.append("")
    for label, pages in sorted(results["found"].items()):
        page_str = str(pages[0]) if len(pages) == 1 else f"AMBIGUOUS {pages}"
        lines.append(f"  {label:<40} p.{page_str}")

    lines.append("")
    lines.append("─" * 70)
    lines.append("NEXT STEP:")
    lines.append(f"  1. Review this log.")
    lines.append(f"  2. Edit {stem}_scan.json if anything needs correction.")
    lines.append(f"  3. Run link_inserter.py only after you are satisfied with the JSON.")
    lines.append("=" * 70)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return json_path, log_path


def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_scan.py \"C:\\path\\to\\file.pdf\"")
        sys.exit(1)

    pdf_path = Path(sys.argv[1])

    if not pdf_path.exists():
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    if pdf_path.suffix.lower() != ".pdf":
        print(f"ERROR: File does not appear to be a PDF: {pdf_path}")
        sys.exit(1)

    print(f"Scanning: {pdf_path.name}")
    print("This may take a moment for large documents...")

    results = scan_pdf(pdf_path)

    json_path, log_path = write_outputs(pdf_path, results)

    stats = results["stats"]
    print("")
    print(f"Done.")
    print(f"  Pages scanned:      {stats['total_pages'] - stats['pages_skipped_no_text']} of {stats['total_pages']}")
    print(f"  Labels found:       {stats['unique_labels_found']}")
    print(f"  Ambiguous labels:   {stats['labels_on_multiple_pages']}")
    print("")
    print(f"Review the log before proceeding:")
    print(f"  {log_path}")
    print(f"  {json_path}")

    if stats["labels_on_multiple_pages"] > 0:
        print("")
        print(f"ACTION NEEDED: {stats['labels_on_multiple_pages']} label(s) found on multiple pages.")
        print("  Open the log file to see which ones, then edit the JSON to resolve them.")


if __name__ == "__main__":
    main()
