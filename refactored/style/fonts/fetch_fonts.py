"""Fetch the fonts (all SIL OFL) used by deckstyle and the slide builder. Run once from this folder.

Open Sans: static TTFs, because matplotlib cannot select weights from the variable font.
Playfair Display (headlines, hero numbers): Google Fonts ships only the variable font, so a static
Regular and Italic instance (wght 400) is cut with fontTools. matplotlib cannot switch OpenType
features, and the deck wants lining numerals (Phase 1, Typography), so in the derived fonts the
digits are remapped through the font's own `lnum` substitution. Because the files are modified, the
family is renamed "Playfair Deck LF" as the OFL requires for Reserved Font Names.
IBM Plex Mono (also OFL) is committed alongside.
"""
from pathlib import Path
from urllib.request import urlretrieve

HERE = Path(__file__).parent
OPENSANS = "https://raw.githubusercontent.com/googlefonts/opensans/main/"
PLAYFAIR = "https://raw.githubusercontent.com/google/fonts/main/ofl/playfairdisplay/"
DECK_FAMILY = "Playfair Deck LF"


def fetch_open_sans():
    for name in ("Regular", "SemiBold", "Light", "Italic"):
        urlretrieve(f"{OPENSANS}fonts/ttf/OpenSans-{name}.ttf", HERE / f"OpenSans-{name}.ttf")
    urlretrieve(f"{OPENSANS}OFL.txt", HERE / "OpenSans-OFL.txt")


def lining_static(var_path, style, out_path):
    """Static wght-400 instance with lining figures as the default digits, renamed family."""
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    font = instancer.instantiateVariableFont(TTFont(var_path), {"wght": 400})
    gsub = font["GSUB"].table
    lookups = {i for fr in gsub.FeatureList.FeatureRecord if fr.FeatureTag == "lnum"
               for i in fr.Feature.LookupListIndex}
    subst = {}
    for i in sorted(lookups):
        for sub in gsub.LookupList.Lookup[i].SubTable:
            sub = getattr(sub, "ExtSubTable", sub)
            subst.update(getattr(sub, "mapping", {}) or {})
    remapped = 0
    for table in font["cmap"].tables:
        for cp, glyph in list(table.cmap.items()):
            if glyph in subst:
                table.cmap[cp] = subst[glyph]
                remapped += 1
    names = font["name"]
    full = f"{DECK_FAMILY} {style}" if style != "Regular" else DECK_FAMILY
    ps = f"{DECK_FAMILY.replace(' ', '')}-{style}"
    for rec in names.names:
        if rec.nameID in (1, 16):
            rec.string = DECK_FAMILY
        elif rec.nameID in (2, 17):
            rec.string = style
        elif rec.nameID == 4:
            rec.string = full
        elif rec.nameID == 6:
            rec.string = ps
        elif rec.nameID == 3:
            rec.string = f"{ps};deck-derived"
    font.save(out_path)
    print(f"{out_path.name}: {remapped} cmap entries -> lining figures")


def fetch_playfair():
    var = {"Regular": "PlayfairDisplay%5Bwght%5D.ttf", "Italic": "PlayfairDisplay-Italic%5Bwght%5D.ttf"}
    for style, remote in var.items():
        local = HERE / f"_PlayfairDisplay-{style}-VF.ttf"
        urlretrieve(PLAYFAIR + remote, local)
        lining_static(local, style, HERE / f"PlayfairDeckLF-{style}.ttf")
        local.unlink()
    urlretrieve(PLAYFAIR + "OFL.txt", HERE / "PlayfairDisplay-OFL.txt")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["opensans", "playfair"]
    if "opensans" in which:
        fetch_open_sans()
    if "playfair" in which:
        fetch_playfair()
    print(sorted(p.name for p in HERE.glob("*.ttf")))
