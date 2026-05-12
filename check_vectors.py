import numpy as np
import json
from pathlib import Path

vec_path = r"C:\Users\piercax\OneDrive - AbbVie Inc (O365)\Desktop 1\testpython\Library\library_vectors.npz"

if not Path(vec_path).exists():
    print("ERROR: library_vectors.npz not found")
    input(); exit()

data = np.load(vec_path, allow_pickle=True)
meta = [json.loads(m) for m in data['metadata']]

matches = [m for m in meta if 'harmonyca' in m.get('text','').lower()]

print(f"Total chunks in vector store: {len(meta)}")
print(f"Chunks containing 'harmonyca': {len(matches)}")

if matches:
    print(f"\nSample match:")
    print(f"  Title: {matches[0]['title']}")
    print(f"  Text preview: {matches[0]['text'][:300]}")
else:
    print("\nNo harmonyca chunks found — vector store was built without full text")
    print("\nSample of what IS in the store:")
    for m in meta[:3]:
        print(f"  Title: {m['title']}")
        print(f"  Text preview: {m['text'][:100]}")
        print()

input("\nPress Enter to exit.")
