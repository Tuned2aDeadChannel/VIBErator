"""
llm_tagger.py
-------------
Step 3 of the PDF pipeline.

Feeds vault notes to a local Ollama model and generates frontmatter suggestions
(type, topic, tags). Writes a staging file for review before anything is applied.

Two modes:

  GENERATE (default):
    Reads notes, calls Ollama, writes a staging file.

    python llm_tagger.py <vault_directory> [options]

    Options:
      --model  <name>    Ollama model to use (default: llama3.1:8b)
      --output <file>    Staging file path   (default: staging.txt in vault directory)
      --all              Re-tag notes that already have all three fields set.
                         Default: skip notes where type, topic, and tags are all present.

  APPLY:
    Reads the staging file (after your review), writes frontmatter to notes.
    Never runs without an explicit --apply flag.

    python llm_tagger.py --apply <staging_file> <vault_directory>

Frontmatter schema (only these three fields are written — title and source are
never touched):
    type:  equipment | reference | supplier | project | note | index
    topic: single lowercase concept
    tags:  [term1, term2, ...]

Ollama must be running locally. Default URL: http://localhost:11434
"""

import re
import sys
import json
import urllib.request
import urllib.error
from pathlib import Path

OLLAMA_URL    = 'http://localhost:11434'
DEFAULT_MODEL = 'llama3.1:8b'
VALID_TYPES   = {'equipment', 'reference', 'supplier', 'project', 'note', 'index'}
MAX_CONTENT_CHARS = 3000   # characters of note body sent to the model


# ---------------------------------------------------------------------------
# Frontmatter helpers
# ---------------------------------------------------------------------------

def read_frontmatter(md_path: Path) -> dict[str, str]:
    """Return frontmatter fields as raw strings, or {} on any failure."""
    try:
        text = md_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return {}
    if not text.startswith('---'):
        return {}
    close = text.find('\n---', 3)
    if close == -1:
        return {}
    result = {}
    for line in text[3:close].splitlines():
        m = re.match(r'^\s*([\w-]+)\s*:\s*(.*)$', line)
        if m:
            result[m.group(1)] = m.group(2).strip()
    return result


def get_note_body(md_path: Path) -> str:
    """Return note content with frontmatter stripped, truncated for the model."""
    try:
        text = md_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError):
        return ''
    if text.startswith('---'):
        close = text.find('\n---', 3)
        if close != -1:
            text = text[close + 4:]
    return text.strip()[:MAX_CONTENT_CHARS]


def needs_tagging(fm: dict[str, str], force: bool) -> bool:
    if force:
        return True
    tags_val = fm.get('tags', '')
    has_tags = bool(tags_val) and tags_val not in ('', '[]')
    return not (fm.get('type') and fm.get('topic') and has_tags)


def _fmt_field(key: str, val) -> str:
    if isinstance(val, list):
        return f'{key}: [{", ".join(val)}]'
    return f'{key}: {val}'


def apply_frontmatter(md_path: Path, updates: dict) -> bool:
    """
    Write type/topic/tags into frontmatter without touching other fields.
    Returns True if the file was changed.
    """
    try:
        text = md_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        print(f'  ERROR reading {md_path.name}: {exc}')
        return False

    if not text.startswith('---'):
        # No frontmatter — prepend it
        new_fm = '\n'.join(['---'] + [_fmt_field(k, v) for k, v in updates.items()] + ['---'])
        new_text = new_fm + '\n' + text
    else:
        close = text.find('\n---', 3)
        if close == -1:
            print(f'  WARNING: Malformed frontmatter in {md_path.name} — skipped.')
            return False

        fm_body    = text[3:close]   # text between opening and closing ---
        body_after = text[close:]    # \n--- onward

        fm_lines = fm_body.splitlines()
        updated_keys: set[str] = set()
        new_lines: list[str] = []

        for line in fm_lines:
            replaced = False
            for key in updates:
                if re.match(rf'^\s*{re.escape(key)}\s*:', line):
                    new_lines.append(_fmt_field(key, updates[key]))
                    updated_keys.add(key)
                    replaced = True
                    break
            if not replaced:
                new_lines.append(line)

        for key, val in updates.items():
            if key not in updated_keys:
                new_lines.append(_fmt_field(key, val))

        new_text = '---' + '\n'.join(new_lines) + body_after

    if new_text == text:
        return False

    try:
        md_path.write_text(new_text, encoding='utf-8')
        return True
    except OSError as exc:
        print(f'  ERROR writing {md_path.name}: {exc}')
        return False


# ---------------------------------------------------------------------------
# Staging file I/O
# ---------------------------------------------------------------------------

def write_staging(results: list[tuple[str, dict]], output_path: Path) -> None:
    lines: list[str] = []
    for file_rel, fields in results:
        lines.append(f'FILE: {file_rel}')
        lines.append(f'  type:  {fields["type"]}')
        lines.append(f'  topic: {fields["topic"]}')
        lines.append(f'  tags:  [{", ".join(fields["tags"])}]')
        lines.append('---')
        lines.append('')
    output_path.write_text('\n'.join(lines), encoding='utf-8')


def parse_staging(staging_path: Path) -> list[tuple[str, dict]]:
    """Parse a staging file. Returns list of (relative_file_path, fields_dict)."""
    try:
        text = staging_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as exc:
        print(f'ERROR: Cannot read staging file: {exc}')
        sys.exit(1)

    entries: list[tuple[str, dict]] = []
    for block in re.split(r'\n---\n?', text):
        block = block.strip()
        if not block:
            continue
        lines = block.splitlines()
        m = re.match(r'^FILE:\s*(.+)$', lines[0])
        if not m:
            continue
        file_rel = m.group(1).strip()
        fields: dict = {}
        for line in lines[1:]:
            fm = re.match(r'^\s*(type|topic|tags)\s*:\s*(.+)$', line)
            if not fm:
                continue
            key = fm.group(1).strip()
            val = fm.group(2).strip()
            if key == 'tags':
                inner = val.strip('[]')
                fields[key] = [t.strip() for t in inner.split(',') if t.strip()]
            else:
                fields[key] = val
        if file_rel and fields:
            entries.append((file_rel, fields))
    return entries


# ---------------------------------------------------------------------------
# Ollama
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """\
Classify the following technical reference note for an Obsidian vault.
Output ONLY these three lines. No explanation, no markdown fences, nothing else.

type: <one of: equipment | reference | supplier | project | note | index>
topic: <single lowercase concept, e.g. seals, fasteners, lubrication, wiring>
tags: [<3 to 8 specific lowercase terms: part numbers, brands, material types, standards>]

Note filename: {filename}
Note content:
{content}"""


def call_ollama(content: str, filename: str, model: str) -> dict | None:
    """Call Ollama and return parsed {type, topic, tags}, or None on failure."""
    prompt = _PROMPT_TEMPLATE.format(filename=filename, content=content)
    payload = json.dumps({
        'model':   model,
        'prompt':  prompt,
        'stream':  False,
        'options': {'temperature': 0.1},
    }).encode('utf-8')

    req = urllib.request.Request(
        f'{OLLAMA_URL}/api/generate',
        data=payload,
        headers={'Content-Type': 'application/json'},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return _parse_response(data.get('response', ''))
    except urllib.error.URLError:
        return None     # caller reports the error with context


def _parse_response(response: str) -> dict | None:
    type_m  = re.search(r'(?m)^type:\s*(\w+)',     response)
    topic_m = re.search(r'(?m)^topic:\s*(.+?)$',   response)
    tags_m  = re.search(r'(?m)^tags:\s*\[(.+?)\]', response)

    if not (type_m and topic_m and tags_m):
        return None

    type_val = type_m.group(1).strip().lower()
    if type_val not in VALID_TYPES:
        type_val = 'note'

    topic_val = topic_m.group(1).strip().lower()
    tags_val  = [t.strip().lower() for t in tags_m.group(1).split(',') if t.strip()]

    return {'type': type_val, 'topic': topic_val, 'tags': tags_val}


def check_ollama(model: str) -> bool:
    """Return True if Ollama is reachable and the model is available."""
    try:
        with urllib.request.urlopen(f'{OLLAMA_URL}/api/tags', timeout=5) as resp:
            data = json.loads(resp.read())
            available = [m['name'] for m in data.get('models', [])]
            if not any(m == model or m.startswith(model.split(':')[0]) for m in available):
                print(f'WARNING: Model "{model}" not found in Ollama.')
                print(f'         Available: {", ".join(available) or "(none)"}')
                print(f'         Run:  ollama pull {model}')
                return False
            return True
    except urllib.error.URLError:
        print(f'ERROR: Cannot reach Ollama at {OLLAMA_URL}')
        print('       Make sure Ollama is running:  ollama serve')
        return False


# ---------------------------------------------------------------------------
# Generate mode
# ---------------------------------------------------------------------------

def run_generate(vault_dir: Path, model: str, output_path: Path, force: bool) -> None:
    if not check_ollama(model):
        sys.exit(1)

    md_files   = sorted(vault_dir.rglob('*.md'))
    to_process = [f for f in md_files if needs_tagging(read_frontmatter(f), force)]
    skipped    = len(md_files) - len(to_process)

    print(f'Vault:   {vault_dir}')
    print(f'Model:   {model}')
    print(f'Notes:   {len(to_process)} to tag  |  {skipped} already complete (use --all to re-tag)')
    print(f'Output:  {output_path}')
    print()

    if not to_process:
        print('Nothing to do.')
        return

    results: list[tuple[str, dict]] = []
    failed:  list[str] = []

    for i, note in enumerate(to_process, 1):
        rel   = note.relative_to(vault_dir)
        label = str(rel)
        print(f'  [{i:>{len(str(len(to_process)))}}/{len(to_process)}] {label}', end='', flush=True)

        body = get_note_body(note)
        if not body:
            print('  — empty, skipped')
            continue

        parsed = call_ollama(body, note.name, model)
        if parsed is None:
            print('  — FAILED (Ollama error or unparseable response)')
            failed.append(label)
            continue

        results.append((label, parsed))
        print(f'  →  {parsed["type"]} / {parsed["topic"]}')

    if not results:
        print()
        print('No results to write.')
        return

    write_staging(results, output_path)

    print()
    print(f'Done.  {len(results)} note(s) tagged, {len(failed)} failed.')
    print()
    print(f'Next steps:')
    print(f'  1. Review {output_path.name}')
    print(f'     Edit any type, topic, or tags that look wrong.')
    print(f'  2. Apply when satisfied:')
    print(f'     python llm_tagger.py --apply "{output_path}" "{vault_dir}"')

    if failed:
        print()
        print(f'Failed notes (not in staging file):')
        for f in failed:
            print(f'  {f}')


# ---------------------------------------------------------------------------
# Apply mode
# ---------------------------------------------------------------------------

def run_apply(staging_path: Path, vault_dir: Path) -> None:
    if not staging_path.exists():
        print(f'ERROR: Staging file not found: {staging_path}')
        sys.exit(1)
    if not vault_dir.is_dir():
        print(f'ERROR: Vault directory not found: {vault_dir}')
        sys.exit(1)

    entries = parse_staging(staging_path)
    if not entries:
        print('Staging file is empty or could not be parsed.')
        sys.exit(0)

    print(f'Staging: {staging_path}')
    print(f'Vault:   {vault_dir}')
    print(f'Entries: {len(entries)}')
    print()

    changed = 0
    missing = 0

    for file_rel, fields in entries:
        note_path = vault_dir / file_rel
        if not note_path.exists():
            print(f'  MISSING  {file_rel}')
            missing += 1
            continue
        modified = apply_frontmatter(note_path, fields)
        status = 'updated' if modified else 'unchanged'
        print(f'  {status:<9} {file_rel}')
        if modified:
            changed += 1

    print()
    print(f'Done.  {changed} note(s) updated, {missing} not found.')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = sys.argv[1:]

    if not args or args[0] in ('-h', '--help'):
        print(__doc__)
        sys.exit(0)

    # Apply mode
    if args[0] == '--apply':
        if len(args) < 3:
            print('Usage: python llm_tagger.py --apply <staging_file> <vault_directory>')
            sys.exit(1)
        run_apply(Path(args[1]), Path(args[2]))
        return

    # Generate mode — parse flags
    vault_dir: Path | None = None
    model       = DEFAULT_MODEL
    output_path: Path | None = None
    force       = False

    i = 0
    while i < len(args):
        if args[i] == '--model' and i + 1 < len(args):
            model = args[i + 1];  i += 2
        elif args[i] == '--output' and i + 1 < len(args):
            output_path = Path(args[i + 1]);  i += 2
        elif args[i] == '--all':
            force = True;  i += 1
        elif not args[i].startswith('--'):
            vault_dir = Path(args[i]);  i += 1
        else:
            print(f'Unknown argument: {args[i]}')
            print('Usage: python llm_tagger.py <vault_directory> [--model NAME] [--output FILE] [--all]')
            sys.exit(1)

    if vault_dir is None:
        print('Usage: python llm_tagger.py <vault_directory> [--model NAME] [--output FILE] [--all]')
        sys.exit(1)
    if not vault_dir.is_dir():
        print(f'ERROR: Directory not found: {vault_dir}')
        sys.exit(1)

    if output_path is None:
        output_path = vault_dir / 'staging.txt'

    run_generate(vault_dir, model, output_path, force)


if __name__ == '__main__':
    main()
