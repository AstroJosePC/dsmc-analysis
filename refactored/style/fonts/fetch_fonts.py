"""Fetch the static Open Sans TTFs (SIL OFL) used by deckstyle. Run once from this folder.

Static files are used because matplotlib cannot select weights from the variable font.
IBM Plex Mono (also OFL) is committed alongside.
"""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://raw.githubusercontent.com/googlefonts/opensans/main/"
HERE = Path(__file__).parent

for name in ("Regular", "SemiBold", "Light", "Italic"):
    urlretrieve(f"{BASE}fonts/ttf/OpenSans-{name}.ttf", HERE / f"OpenSans-{name}.ttf")
urlretrieve(f"{BASE}OFL.txt", HERE / "OpenSans-OFL.txt")
print(sorted(p.name for p in HERE.glob("OpenSans*")))
