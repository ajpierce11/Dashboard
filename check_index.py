import json
from pathlib import Path

index_path = r"C:\Users\piercax\OneDrive - AbbVie Inc (O365)\Desktop 1\testpython\Library\library_index.json"
product = "harmonyca"

with open(index_path) as f:
    data = json.load(f)

entries = [e for e in data["entries"] if not e.get("deleted")]
matches = [
    e for e in entries
    if product in (e.get("title","") + e.get("full_text","") + e.get("preview","")).lower()
]

print(f"Total entries: {len(entries)}")
print(f"Entries mentioning '{product}': {len(matches)}")

# Check if full_text field exists at all in first entry
first = entries[0] if entries else {}
print(f"\nFields in first entry: {list(first.keys())}")
print(f"'full_text' field exists: {'full_text' in first}")
print()

for m in matches[:5]:
    print(f"  {m['title']}")
    print(f"    preview: {len(m.get('preview',''))} chars")
    print(f"    full_text: {len(m.get('full_text',''))} chars")
    print()

input("Press Enter to exit.")