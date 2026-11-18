"""Review PDF for the slide 3 prototype: FIG 3 with MAVEN's orbit 2521 pass (FIG 3m), full slide + builds.

    cd refactored && conda run -n dsmc python deck/cache_slide03_maven.py      # when a source changes
    cd refactored && conda run -n dsmc python deck/fig03m_regimes_maven.py
    cd refactored && conda run -n dsmc python slides/build_slide03_maven_preview.py

Uses the chrome, audits (style module, figure size / fonts / provenance), collision check, composition and
build-jitter check of build_deck.py unchanged; the deck PDFs are not touched. Slide 3 is laid out as in
build_deck.slide3() with FIG 3m in FIG 3's slot and one more build step:
  1  regime sector (FIG 3m step 1 = FIG 3) + WHERE FLUIDS FAIL card
  2  MAVEN orbit 2521 pass, spacecraft and label
  3  Navier-Stokes statement
  4  heating / dissipation statement
Page 1 of the output is the full slide; pages 2-5 are build steps 1-4.

Outputs: slides/build/slide03_maven_trajectory_preview.pdf, slides/build/slide03_maven_preview/page-N.png
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_deck as bd  # noqa: E402
from build_deck import (EVID_BOTTOM, EVID_X, EVID_Y, SANS, WORD_W, WORD_X, Card, Slide, cite, ds,  # noqa: E402
                        footer, fp, header)

STEM = "fig03m_regime_maven"
bd.SIZE_KEY[STEM] = "evidence"                    # FIG 3's slot
OUT_PDF = bd.OUT / "slide03_maven_trajectory_preview.pdf"
PREVIEW = bd.OUT / "slide03_maven_preview"
WORK = bd.OUT / "_work_slide03_maven"
# the untracked style spec lives only in the main checkout; audit against it when it is there
MAIN_DOCS = Path("/Volumes/MyBook/Education and Research/Research/DSMC Exosphere GWs/dsmc_analyzer/refactored/docs")


def slide3m():
    s = Slide(3)
    header(s, "WHY KINETIC",
           "Near the exobase, gas becomes rarefied and mean free paths grow, "
           "so non-equilibrium solutions require kinetic models.")
    s.reveal(1)                                   # build: regimes and Kn, MAVEN's pass, then the statements
    s.place(f"{STEM}_deck", EVID_X, EVID_Y, builds=1)
    body = fp(SANS, 20)
    y = EVID_Y + 8
    for k, stmt in enumerate((
        r"Navier-Stokes errors become significant when local $\mathrm{Kn}>0.1$, "
        r"and continuum models must be replaced by molecular at $\mathrm{Kn}\sim0.2$.",
        "Heating and dissipation effects self-consistently emerge from collisions.",
    )):
        s.reveal(3 + k)
        y = s.paragraph(WORD_X, y, stmt, body, WORD_W, 26) + 37
    s.reveal(1)
    card = Card("WHERE FLUIDS FAIL",
                eq=[[(r"$\mathrm{Kn} = \lambda_{\mathrm{mfp}}/H \;\to\; 1$", fp(SANS, 20), ds.MIDNIGHT),
                     ("  at the exobase", fp(SANS, 16, 300), ds.GRANITE)]],
                lines=[(r"$\lambda_{\mathrm{mfp}}$: distance between collisions", 16, ds.GRANITE, 300),
                       ("H: scale height (40 km here)", 16, ds.GRANITE, 300)])
    _, ch = card.layout()
    card.draw(s, WORD_X, EVID_BOTTOM - ch)
    footer(s, cite("Bird 1994; Leclercq et al. 2020; Tian et al. 2023; MAVEN NGIMS + KP, orbit 2521"))
    return s


def main():
    if MAIN_DOCS.exists():
        bd.DOCS = MAIN_DOCS
    for note in bd.audit_style():
        print(note)
    commit = subprocess.run(["git", "-C", str(bd.REFACTORED), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    s = slide3m()
    stems = sorted({l[0] for l in s.layers})
    for stem in stems:
        bd.audit_figure(stem)
    for k in s.pages():
        bd.check_layout(s, k)
    steps = list(s.pages())
    print(f"figures {', '.join(stems)}: size {ds.SIZES['evidence']} in, deck fonts, provenance OK; "
          f"layout: no clipping or collisions on the full slide and build steps {steps}")
    pages = [(s, s.n_steps)] + [(s, k) for k in steps]
    bd.compose(pages, commit, OUT_PDF, work=WORK,
               subject=f"Slide 3 prototype (MAVEN orbit 2521 on FIG 3): full slide + build steps "
                       f"{steps[0]}-{steps[-1]}; figures/deck @ {commit}")
    n, fonts = bd.verify(len(pages), OUT_PDF, PREVIEW, name="page")
    print(f"{OUT_PDF.relative_to(bd.REFACTORED)}: {n} pages, 960 x 540 pt; fonts: {', '.join(fonts)}")
    # build jitter: every build page against page 1 (the full slide)
    full_dir, prog_dir = PREVIEW / "_full", PREVIEW / "_prog"
    for d in (full_dir, prog_dir):
        d.mkdir()
    pngs = sorted(PREVIEW.glob("page-*.png"))
    shutil.copy(pngs[0], full_dir / pngs[0].name)
    for p in pngs[1:]:
        shutil.copy(p, prog_dir / p.name)
    worst = bd.check_builds([(s, k) for k in steps], full_dir, prog_dir, {id(s): 0})
    shutil.rmtree(full_dir)
    shutil.rmtree(prog_dir)
    print(f"builds: slide 3 x{len(steps)}; no element moves between steps "
          f"(worst mismatch {100 * worst:.3f}% of inked pixels)")
    shutil.rmtree(WORK)


if __name__ == "__main__":
    main()
