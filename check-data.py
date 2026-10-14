import re
from pathlib import Path
from collections import Counter

root = Path("data/NEU-DET")

for split in ["train", "validation"]:
    files = list((root / split / "images").rglob("*.jpg"))
    classes = [re.sub(r"_\d+$", "", f.stem) for f in files]
    print(split, len(files), "images")
    print(Counter(classes))