# Graphite Telemetry: one species, four boundaries

*A design philosophy for instruments that report what they measured, and nothing else. This version is adapted
to a report whose subject is one species, H₂, under four upper-boundary conditions, compared with a published
reference.*

## Manifesto

**Graphite Telemetry** treats the page as an instrument readout. Everything is graphite, like the ink of a sharp pencil on calibrated paper. Charcoal marks what is measured. Mid-greys mark what is inferred. The palest greys are scaffolding: grids, guides and the lines that help the eye find a value without competing with it. This report has one subject, H₂, so it admits exactly one ink, H₂ orange. That ink is not decoration. It marks the subject of the report, the ballistic treatment of particles above the upper boundary. When orange appears it means *a ballistic hold*, and nothing else.

**The boundary condition is told by stroke, not by hue.** Four upper-boundary conditions share the page. They are distinguished the way a draughtsman separates lines on a single plate: by weight, dash and marker, never by a second colour.
- **Reflect** is the reference against which everything is compared. It is drawn in charcoal: a solid line with hollow circles.
- **The Kepler hold** is the ballistic treatment in production use. It is a solid orange line with filled squares.
- **Explicit RK4 tracking** is the same physics computed another way. It shares the orange of the Kepler hold, with a dashed line and hollow diamonds. When the two agree, the dashed line lies on the solid one. That overlap is the verification result, made visible.
- **Free escape** is the limiting case. It is a grey dotted line with hollow triangles, recessive because it is the bound and not the subject.

A reader who learns this key once can read every panel without the legend.

**Theory and literature are drawn quieter than measurement.** The collisionless solution is the analytic exosphere computed from the same source parameters. It is a thin grey dashed line throughout: the baseline the DSMC departs from. Values from Carberry Mogan et al. (2021) are digitised from another body, another species setup and another time normalisation. They appear only as small graphite crosses: present, legible, and plainly not ours. No line joins them, because a line would claim a continuity we did not measure. Where a figure puts their axis beside ours (*t*/*t*<sub>orb</sub> beside *t*/τ<sub>90</sub>), the axis label states both, so the comparison is offered and never hidden.

**Missing data is shown as missing.** Several inputs are still running while this report is assembled. A pending result is never left as a blank space. It is drawn as a dashed light-grey box that names what is missing and when it will arrive. In the text it reads `[pending: reason]` in grey monospace. The page stays honest at every stage of the campaign and can be regenerated to completion without any edit by hand.

**Space is the primary material.** Composition follows a strict modular grid with wide, deliberate margins. Panels share axes where they share meaning: radial distance in *r*/*r*₀ on a log axis, and time on a log axis spanning the hold's filling. Gutters are identical. Nothing touches the frame, nothing overlaps, and every label sits in its own clear air.

**Line weight carries hierarchy, so colour does not have to.**
- Axes are 0.6 pt mid-grey hairlines.
- References are 0.9 pt grey dashes.
- Measurements are the heaviest strokes. Their uncertainty is drawn as thin whiskers or as a band in the stroke's own ink at low opacity.
- Exobase markers are thin charcoal lines: dashed for Kn = 1 and dash-dotted for Kn₀ = 1, matching the convention of the paper being compared.

**Typography is quiet, exact and mechanical.** A clean grotesque sans carries prose and labels. Every number is set in a monospaced face and comes from a generated file, never typed by hand. Units follow in plain small type. Titles state what is plotted and do not narrate.

**The whole is a single calibrated system.** Every figure uses the same palette, line weights, type sizes and UBC key, so the five figures read as one instrument with many channels. Tables follow the same law: three horizontal rules, no verticals, numbers right-aligned in monospace.

---

## Implementation notes (this report)

- **Inks.**
  - Charcoal `#1f1f1d`: measurements, body text, the reflect UBC.
  - Graphite `#4a4a47`: secondary text, Carberry Mogan et al. (2021) markers.
  - Grey `#7a7a76`: collisionless references, free escape.
  - Light grey `#b9b8b3`: axes, ticks, pending boxes.
  - Rule grey `#e6e5e1`: grid lines and table rules.
  - Paper `#ffffff`.
- **Subject ink.** H₂ `#eb6834`, used for the ballistic UBCs (Kepler hold and explicit RK4) only. Bands are the same ink at 18% opacity. The O ink of the template (`#2a78d6`) is defined in the LaTeX preamble for consistency but is not used.
- **UBC key** (`scripts/style.py`, `UBC`):

  | UBC | Ink | Line | Marker |
  |---|---|---|---|
  | 1 reflect | charcoal | solid, 1.4 pt | hollow circle |
  | 2 Kepler hold | H₂ orange | solid, 1.6 pt | filled square |
  | 5 explicit RK4 | H₂ orange | dashed (4, 1.8), 1.6 pt | hollow diamond |
  | 3 free escape | grey | dotted (1, 1.4), 1.3 pt | hollow triangle |

- **References.** The collisionless analytic solution (`dsmclib.exosphere`) is a grey dashed line (5, 2.5). The paper's values are graphite crosses with no connecting line, labelled "digitised" in captions. An *r*<sub>Hill</sub> arrow marks the Mars Hill radius when it lies off the axis.
- **Pending.** In figures, `style.pending()` draws a dashed light-grey box with grey text. In LaTeX, `\pending{reason}` prints `[pending: reason]` in grey monospace. In wide tables it is shortened locally to `[pending]`, and the caption gives the reason.
- **Type.**
  - Instrument Sans for text and labels.
  - JetBrains Mono for numbers and tick labels.
  - Fira Math for equations.
  - Figure text is 8–9 pt at final size. Panel titles are 9 pt, flush left, with a lower-case panel letter.
- **Strokes.**
  - Axes 0.6 pt.
  - References 0.9 pt.
  - Data 1.3–1.8 pt.
  - Raw traces 0.5 pt at 45% opacity.
  - Markers 3.5–4.5 pt.
- **Geometry.**
  - Full-width figures are 6.5 in wide.
  - Radial axes are *r*/*r*₀ on a log scale, with *r*₀ at 100 km altitude.
  - Time axes are log scaled in seconds, or in *t*/τ<sub>90</sub> where the paper's *t*/*t*<sub>orb</sub> is overlaid.
- **Export.** Every figure is written as vector PDF for LaTeX and as 300-dpi PNG for review (`scripts/make_figures.py`).
