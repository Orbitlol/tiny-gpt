import re
import pathlib

RAW_DIR = pathlib.Path("data/raw")
OUT_PATH = pathlib.Path("data/processed/corpus.txt")

def clean_markdown(text: str) -> str:
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[([^\]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\*\-\+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[*_]{1,3}([^*_]+)[*_]{1,3}', r'\1', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.md"))
    assert files, f"No .md files found in {RAW_DIR}"
    chunks = [clean_markdown(f.read_text(encoding="utf-8", errors="ignore")) for f in files]
    OUT_PATH.write_text("\n\n".join(chunks), encoding="utf-8")
    print(f"Wrote {OUT_PATH} ({OUT_PATH.stat().st_size/1024:.1f} KB) from {len(files)} files")

if __name__ == "__main__":
    main()
