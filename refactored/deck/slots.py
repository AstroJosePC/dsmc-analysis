"""Placed sizes (inches) of review-only deck figures that do not use a deckstyle SIZES slot.

These are layout slots, not style tokens: each is cut from the 13.333 x 7.5 in slide grid for one
alternative slide in slides/alternatives_review.pdf, and the figure is drawn at exactly this size so
nothing is rescaled when placed. slides/build_deck.py audits every placed PDF against them.

Adopted slots live in deckstyle.SIZES instead: fig01 title art -> "title", 4D -> "plate",
6A wedge -> "evidence_tall", 6A plots -> "column" (STYLE_VERSION 1.1); 7B-2 strip -> "strip" (1.2).
If another alternative is adopted, promote its slot the same way (with a STYLE_VERSION bump).

Slide geometry they are cut from (points, y down; build_deck.py): margins 40, content 160-500,
word column 4.67 in, gutter 0.33 in.
"""
SLOTS = {
    # slide 3C: full-width figure over a row of three callouts
    "fig03c_collision_clock": (12.22, 3.00),
    # slide 4B: full-width figure over a row of three callouts
    "fig04b_powers_of_ten": (12.22, 3.00),
    # slide 7A: full-width zoom strip (full ring -> wheel sector -> polar band) over three callouts
    "fig07b_zoom_strip": (12.22, 3.10),
    # slide 6B: full-width wedge band over three plots with hero numbers
    "fig06a_s6_wedge_band": (12.22, 2.30),
    "fig06b_latt_fit_row": (3.87, 1.80),
    "fig06c_lambdah_fit_row": (3.87, 1.80),
    "fig06d_latt_profile_row": (3.87, 1.80),
}

# Adopted figures drawn at a deckstyle SIZES key (stem -> key).
ADOPTED = {"fig01_title_wedge": "title", "fig04d_scales_and_step": "plate",
           "fig06a_s6_halfwedge": "evidence_tall", "fig06b_latt_fit_col": "column",
           "fig06c_lambdah_fit_col": "column", "fig07e_sector_strip": "strip"}


def size_in(stem: str, sizes: dict) -> tuple[float, float]:
    """Placed size of a figure stem: deckstyle SIZES (pass ds.SIZES) if adopted, else its slot."""
    stem = stem.removesuffix("_deck")
    return sizes[ADOPTED[stem]] if stem in ADOPTED else SLOTS[stem]
