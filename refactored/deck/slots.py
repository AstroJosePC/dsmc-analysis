"""Placed sizes (inches) of deck figures that do not use a deckstyle SIZES slot.

These are layout slots, not style tokens: each is cut from the 13.333 x 7.5 in slide grid for one
slide composition, and the figure is drawn at exactly this size so nothing is rescaled when placed.
The figure scripts draw at these sizes and slides/build_deck.py audits every placed PDF against
them. If an alternative slide is adopted, promote its slot into deckstyle.SIZES (with a
STYLE_VERSION bump) rather than keeping it here.

Slide geometry they are cut from (points, y down; build_deck.py): margins 40, content 160-500,
word column 4.67 in, gutter 0.33 in.
"""
SLOTS = {
    # slide 1: the right field of the dark title slide
    "fig01_title_wedge": (7.6, 6.4),
    # slide 3C: full-width figure over a row of three callouts
    "fig03c_collision_clock": (12.22, 3.00),
    # slide 4B: full-width figure over a row of three callouts
    "fig04b_powers_of_ten": (12.22, 3.00),
    # slide 6A: tall evidence field (to the footer) beside two stacked plots with hero numbers
    "fig06a_s6_halfwedge": (7.22, 4.55),
    "fig06b_latt_fit_col": (4.67, 1.62),
    "fig06c_lambdah_fit_col": (4.67, 1.62),
    # slide 6B: full-width wedge band over three plots with hero numbers
    "fig06a_s6_wedge_band": (12.22, 2.30),
    "fig06b_latt_fit_row": (3.87, 1.80),
    "fig06c_lambdah_fit_row": (3.87, 1.80),
    "fig06d_latt_profile_row": (3.87, 1.80),
}
