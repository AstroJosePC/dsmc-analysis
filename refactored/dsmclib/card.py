"""Simulation card: a one-page PDF (+ JSON) audit sheet for every run with a run file (<alias>_run.nc).

    python -m dsmclib.card OH2_A                         # one run by registry alias
    python -m dsmclib.card --campaign oh2_escape_rerun   # every run of a campaign that has a run file
    python -m dsmclib.card --all                         # every registered run that has a run file
    python -m dsmclib.card --run-file /path/Results/X_run.nc # any run file, registered or not
    options: --overwrite (re-render existing cards)

    from dsmclib.card import extract, make_card
    card = extract("OH2_A")          # dict: the card's data, nothing rendered
    make_card("OH2_A")               # JSON + PDF

Everything on the card comes from the run file (root attributes incl. every param_* line of
Simulation_Parameters, /macroscopic, /rates_surface, /rates_escape, /info_surface) plus the run
folder's run_meta.txt, COMPLETE.ascii, Input/ and DSMC binary (hashed). Nothing is typed by hand.

Outputs (new files only; existing cards are kept unless overwrite=True):
    <Results>/<alias>_card.pdf, <Results>/<alias>_card.json      next to the run file
    outputs/cards/<alias>_card.{pdf,json} and outputs/cards/cards.csv   central copy + index

Rendering: matplotlib (Graphite Telemetry, dsmclib.graphite) for the domain ruler and the
inventory panels, LaTeX compiled with tectonic, one US Letter page (enforced).
"""

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__ as DSMCLIB_VERSION
from .constants import AMU
from .ncrun import RUN_SUFFIX, find_run_file
from .paths import OUTPUT_ROOT

CARD_VERSION = "1.1"  # 1.1: run-file naming, npart keys, tier/absent_params status (convention 1.0)
CARD_SUFFIX = "_card"
CENTRAL = OUTPUT_ROOT / "cards"
UBC_NAMES = {"UBC_FREE_ESCAPE": "free escape", "UBC_SPECULAR": "specular", "UBC_DIFFUSE": "diffuse"}


def ubc_name(code: str) -> str:
    """Readable upper-boundary condition: UBC_FREE_ESCAPE -> free escape, UBC_BALLISTIC -> ballistic."""
    code = str(code or "").strip()
    return UBC_NAMES.get(code, code.removeprefix("UBC_").replace("_", " ").lower() or "n/a")


# =============================================================================== extraction
def _num(text) -> Optional[float]:
    """First number in a param string such as '3389.5000000000000      [km]'."""
    if text is None:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d*)?(?:[EeDd][-+]?\d+)?", str(text))
    return float(m.group(0).replace("D", "E").replace("d", "e")) if m else None


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _sha256_tree(folder: Path) -> tuple[Optional[str], int]:
    """Hash of a folder: sha256 over sorted 'relative/path<TAB>file-sha256' lines."""
    if not folder.is_dir():
        return None, 0
    files = sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.startswith("."))
    h = hashlib.sha256()
    for p in files:
        h.update(f"{p.relative_to(folder).as_posix()}\t{_sha256(p)}\n".encode())
    return h.hexdigest(), len(files)


def _run_meta(run_dir: Path) -> dict:
    p = run_dir / "run_meta.txt"
    if not p.is_file():
        return {}
    return dict(re.findall(r"^(\w+):\s+(.*?)\s*$", p.read_text(), re.M))


def _registry_match(results: Path):
    """Registered run whose Results folder is `results` (None if unregistered or no registry)."""
    try:
        from . import registry

        for r in registry.all_runs():
            try:
                if r.path.resolve() == results.resolve():
                    return r
            except OSError:
                continue
    except Exception:
        return None
    return None


def _cell_volumes(macro) -> np.ndarray:
    """Cell volumes (r, zen, azi) of a spherical grid from the bounds (angles in degrees)."""
    rb = np.asarray(macro.variables["radial_bounds"][:], float)
    zb = np.asarray(macro.variables["zenith_bounds"][:], float)
    ab = np.asarray(macro.variables["azimuthal_bounds"][:], float)
    if np.nanmax(np.abs(zb)) > 2 * np.pi + 0.1 or np.nanmax(np.abs(ab)) > 2 * np.pi + 0.1:
        zb, ab = np.radians(zb), np.radians(ab)
    vr = (rb[:, 1] ** 3 - rb[:, 0] ** 3) / 3.0
    vz = np.cos(zb[:, 0]) - np.cos(zb[:, 1])
    va = ab[:, 1] - ab[:, 0]
    return np.abs(vr[:, None, None] * vz[None, :, None] * va[None, None, :])


def _boundary_total(group, var: str, n_species: int) -> list:
    """Total computational particles through a boundary: sum over cells and outputs of rate x window length."""
    if group is None or var not in group.variables:
        return [None] * n_species
    rate = np.asarray(group.variables[var][:], float)  # (species, time, cell) [s-1]
    t = np.asarray(group.variables["time"][:], float)
    win = np.diff(np.concatenate([[0.0], t]))
    return [float(np.nansum(rate[s] * win[:, None])) for s in range(n_species)]


def _collisions(attrs: dict) -> list:
    out, k = [], 1
    while True:
        sfx = "" if k == 1 else f"_{k}"
        if f"param_Collision_Type{sfx}" not in attrs:
            break
        out.append({"type": int(_num(attrs[f"param_Collision_Type{sfx}"]) or 0),
                    "species_1": str(attrs.get(f"param_Species_1{sfx}", "")).strip(),
                    "species_2": str(attrs.get(f"param_Species_2{sfx}", "")).strip(),
                    "reduced_mass_amu": _num(attrs.get(f"param_Reduced_Mass{sfx}"))})
        k += 1
    return out


def _range(values) -> list:
    v = np.asarray(values, float).ravel()
    v = v[np.isfinite(v)]
    return [float(v.min()), float(v.max())] if v.size else [None, None]


def extract(target, alias: Optional[str] = None) -> dict:
    """The card's data for an alias, a Results/run folder or a run-file path (no rendering)."""
    import netCDF4

    rf, alias0 = find_run_file(target)
    results = rf.parent
    run_dir = results.parent
    reg = _registry_match(results)
    alias = alias or alias0 or (reg.alias if reg else None) or rf.name[: -len(RUN_SUFFIX)]

    ds = netCDF4.Dataset(rf)
    try:
        A = {k: ds.getncattr(k) for k in ds.ncattrs()}
        M = ds.groups["macroscopic"]
        S = ds.groups.get("rates_surface")
        E = ds.groups.get("rates_escape")
        I = ds.groups.get("info_surface")
        names = [s.strip() for s in str(A.get("species_names", "")).split(",") if s.strip()]
        ns = len(names)
        R = float(A["planet_radius"])
        dt = float(A["dt"])
        n_steps = int(A["n_steps"])
        avg_steps = int(A.get("avg_steps", 0))

        # ---- domain
        rb = np.asarray(M.variables["radial_bounds"][:], float)
        edges = np.concatenate([rb[:, 0], rb[-1:, 1]])
        dr = rb[:, 1] - rb[:, 0]
        n_rad, n_zen, n_azi = (len(M.dimensions[d]) for d in ("radial", "zenith", "azimuthal"))
        zb = np.asarray(M.variables["zenith_bounds"][:], float)
        ab = np.asarray(M.variables["azimuthal_bounds"][:], float)
        ubc = str(A.get("upper_boundary_condition", A.get("param_Upper_Radial_Boundary_Condition", ""))).strip()
        domain = {
            "planet_radius_km": R / 1e3, "planet_mass_kg": float(A["planet_mass"]),
            "lower_alt_km": (float(A["lower_boundary_radius"]) - R) / 1e3,
            "upper_alt_km": (float(A["upper_boundary_radius"]) - R) / 1e3,
            "ubc": ubc, "n_radial": n_rad, "n_zenith": n_zen, "n_azimuthal": n_azi,
            "n_cells": n_rad * n_zen * n_azi,
            "dr_min_km": float(dr.min() / 1e3), "dr_max_km": float(dr.max() / 1e3),
            "zenith_deg": [float(zb.min()), float(zb.max())], "azimuth_deg": [float(ab.min()), float(ab.max())],
            "edge_alt_km": ((edges - R) / 1e3).round(4).tolist(),
        }

        # ---- time
        t = np.asarray(M.variables["time"][:], float)
        win = np.asarray(M.variables["avg_time_interval"][:], float) if "avg_time_interval" in M.variables \
            else np.full(t.size, dt * avg_steps)
        expected_out = n_steps // avg_steps if avg_steps else None
        time = {"dt_s": dt, "n_steps": n_steps, "t_end_s": float(t[-1]), "t_nominal_s": n_steps * dt,
                "samp_steps": int(A.get("samp_steps", 0)), "avg_steps": avg_steps,
                "avg_window_s": float(np.median(win)), "n_outputs": int(t.size), "expected_outputs": expected_out}

        # ---- forcing
        start, end = int(A.get("wave_start_step", 0) or 0), int(A.get("wave_end_step", 0) or 0)
        omega, eps = float(A.get("wave_omega", 0) or 0), float(A.get("wave_amplitude", 0) or 0)
        in_run = omega > 0 and eps > 0 and 0 < start < n_steps
        sim_end = min(end, n_steps) if end > 0 else n_steps
        forcing = {
            "type": str(A.get("param_Wave_Type", "")).strip() or None, "amplitude": eps, "omega_rad_s": omega,
            "period_s": 2 * np.pi / omega if omega > 0 else None,
            "start_step": start, "start_time_s": start * dt, "end_step": end, "end_time_s": end * dt,
            "in_run": bool(in_run), "end_beyond_run": bool(end > n_steps),
            "periods_configured": (end - start) * dt * omega / (2 * np.pi) if omega > 0 and end > start else None,
            "periods_in_run": (sim_end - start) * dt * omega / (2 * np.pi) if in_run else 0.0,
            "horizontal_wavelength_deg": float(A.get("wave_horizontal_wavelength_deg", 0) or 0),
            "injection_width_cells": int(A.get("wave_injection_width_cells", 0) or 0),
        }

        # ---- populations (chunked over time so 3-D runs stay in memory)
        num_part_v = M.variables["num_part"]  # (species, time, radial, zenith, azimuthal)
        per_t = ns * n_rad * n_zen * n_azi
        chunk = max(1, int(2e7 // max(per_t, 1)))
        npart = np.zeros((ns, t.size))
        for i in range(0, t.size, chunk):
            npart[:, i:i + chunk] = np.asarray(num_part_v[:, i:i + chunk], float).sum(axis=(2, 3, 4))
        k_tail = max(1, t.size // 10)
        tail_bottom = np.asarray(num_part_v[:, t.size - k_tail:, 0], float).mean(axis=1)  # (species, zen, azi)
        tail_top = np.asarray(num_part_v[:, t.size - k_tail:, n_rad - 1], float).mean(axis=1)
        last_n = np.asarray(M.variables["num_dens"][:, t.size - 1], float)  # (species, r, z, a)
        last_p = np.asarray(num_part_v[:, t.size - 1], float)
        V = _cell_volumes(M)
        W = np.asarray(M.variables["particle_weight"][:], float)
        mass = np.asarray(M.variables["species_mass"][:], float)
        nphys_last = (last_n * V[None]).reshape(ns, -1).sum(axis=1)
        ok = last_p > 0
        dev = np.abs(last_n * V[None] / (W[:, None, None, None] * np.where(ok, last_p, 1.0)) - 1.0)
        nw_dev = float(dev[ok].max()) if ok.any() else None
        inj = _boundary_total(S, "num_part_inject", ns)
        ret = _boundary_total(S, "num_part_return", ns)
        esc = _boundary_total(E, "num_part_escape", ns)
        stride = max(1, t.size // 600)

        # ---- lower boundary (Info/Surface_Cells)
        def info(var, s):
            if I is None or var not in I.variables:
                return [None, None]
            v = np.asarray(I.variables[var][:], float)
            return _range(v[s] if v.ndim > 1 else v)

        species = []
        for s, nm in enumerate(names):
            species.append({
                "name": nm, "mass_amu": float(mass[s] / AMU), "weight": float(W[s]),
                "n0_m3": info("num_dens", s), "T0_K": info("T0", s), "speed_m_s": info("speed", s),
                "flux_m2_s": info("flux", s), "flow_rate_s": info("flow_rate", s),
                "jeans_lambda0": info("jeans_parameter", s), "jeans_rate0_s": info("jeans_escape_rate", s),
                "npart_final": float(npart[s, -1]), "npart_peak": float(npart[s].max()),
                "nphys_final": float(nphys_last[s]),
                "npart_bottom_cell_tail": _range(tail_bottom[s]), "npart_top_cell_tail": _range(tail_top[s]),
                "npart_injected": inj[s], "npart_returned": ret[s], "npart_escaped": esc[s],
            })
        populations = {"time_s": t[::stride].round(3).tolist(),
                       "npart": {nm: npart[s, ::stride].round(1).tolist() for s, nm in enumerate(names)},
                       "tail_fraction": k_tail / t.size}
        collisions = _collisions(A)
        history = str(A.get("history", ""))
    finally:
        ds.close()

    # ---- provenance
    meta = _run_meta(run_dir)
    commit_raw = meta.get("commit", "")
    started = str(A.get("run_started", meta.get("started", ""))) or None
    finished = str(A.get("run_finished", meta.get("finished", ""))) or None
    wall = None
    try:
        wall = (datetime.fromisoformat(finished) - datetime.fromisoformat(started)).total_seconds()
    except (TypeError, ValueError):
        pass
    input_sha, n_input = _sha256_tree(run_dir / "Input")
    provenance = {
        "run_config": str(A.get("run_config", meta.get("config", ""))) or None,
        "commit": str(A.get("source_commit", commit_raw.split(" ")[0] if commit_raw else "")) or None,
        "branch": str(A.get("source_branch", commit_raw.split("(")[-1].rstrip(")") if "(" in commit_raw else "")) or None,
        "host": str(A.get("run_host", meta.get("host", ""))) or None,
        "mpi_ranks": int(_num(A.get("mpi_ranks", meta.get("np"))) or 0) or None,
        "started": started, "finished": finished, "wall_s": wall,
        "builder_version": str(A.get("builder_version", "")) or None, "run_file_history": history or None,
        "strict_mode": str(A.get("strict_mode", "")).lower() == "true",
        "dsmclib_version": DSMCLIB_VERSION, "card_version": CARD_VERSION,
        "sha256": {"run_file": _sha256(rf), "input": input_sha, "run_meta": _sha256(run_dir / "run_meta.txt"),
                   "executable": _sha256(run_dir / "DSMC")},
        "n_input_files": n_input,
        "paths": {"run_dir": str(run_dir), "run_file": str(rf)},
        "data_tier": str(A.get("data_tier", "")) or None,
        "convention_version": str(A.get("convention_version", "")) or None,
        "absent_params": [x for x in str(A.get("absent_params", "")).split(",") if x],
    }

    # ---- status
    def st(level, label, detail=""):
        return {"level": level, "label": label, "detail": detail}

    status = []
    if (run_dir / "COMPLETE.ascii").is_file():
        status.append(st("ok", "completed", "COMPLETE.ascii"))
    elif meta and "finished" not in meta:
        status.append(st("warn", "not finished", "no COMPLETE.ascii or finish time: running or aborted; the run file is a snapshot"))
    else:
        status.append(st("warn", "no completion marker", "COMPLETE.ascii missing"))
    if expected_out is not None:
        status.append(st("ok" if t.size == expected_out else "warn", "outputs", f"{t.size:,} / {expected_out:,}"))
    dt_out = np.diff(t)
    uniform = dt_out.size == 0 or float(np.ptp(dt_out)) <= 1e-6 * max(float(np.median(dt_out)), 1e-12)
    reach = abs(time["t_end_s"] - time["t_nominal_s"]) <= dt * max(avg_steps, 1)
    status.append(st("ok" if uniform and reach else "warn", "time axis",
                     ("uniform" if uniform else "non-uniform") + (", ends at the last step" if reach else ", ends before the last step")))
    if nw_dev is not None:
        status.append(st("ok" if nw_dev < 1e-6 else "warn", "N_phys = W·N_part", f"max dev {nw_dev:.0e}"))
    status.append(st("ok" if provenance["strict_mode"] else "warn", "strict build",
                     f"builder {provenance['builder_version']}"))
    if provenance["data_tier"] == "run":
        status.append(st("ok", "run file", f"convention {provenance['convention_version']}"))
    else:
        status.append(st("warn", "pre-convention run file", "no data_tier attribute (builder < 1.2.0)"))
    if provenance["absent_params"]:
        status.append(st("note", "solver lacks params", f"{len(provenance['absent_params'])} typed keys absent"))
    if not meta:
        status.append(st("warn", "no run_meta.txt", "provenance from the run file only"))
    if forcing["in_run"] and forcing["end_beyond_run"]:
        status.append(st("note", "forcing end beyond run end",
                         f"{forcing['periods_in_run']:.2f} of {forcing['periods_configured']:.1f} periods simulated"))
    elif not forcing["in_run"] and forcing["omega_rad_s"] > 0 and forcing["amplitude"] > 0:
        status.append(st("note", "unforced run", "forcing configured beyond run end"))

    return {
        "card_version": CARD_VERSION, "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run": {"alias": alias, "campaign": reg.campaign if reg else None, "role": reg.role if reg else None,
                "notes": reg.notes if reg else None, "species": names},
        "status": status, "domain": domain, "time": time, "forcing": forcing, "species": species,
        "collisions": collisions, "populations": populations, "provenance": provenance,
    }


# =============================================================================== LaTeX helpers
_TEX_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
            "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def tex(s) -> str:
    return "".join(_TEX_ESC.get(c, c) for c in str(s))


def tex_path(s) -> str:
    """Monospaced path that may break after / and _."""
    return r"\texttt{" + tex(s).replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}") + "}"


def tex_species(name: str) -> str:
    return re.sub(r"(\d+)", r"\\textsubscript{\1}", tex(name))


def num(x, d=0) -> str:
    if x is None or not np.isfinite(x):
        return r"\na"
    s = f"{x:,.{d}f}".replace(",", r"\,").replace("-", "\u2212")
    return rf"\num{{{s}}}"


def sci(x, d=2) -> str:
    if x is None or not np.isfinite(x):
        return r"\na"
    if x == 0:
        return r"\num{0}"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    if round(abs(m), d) >= 10:
        m, e = m / 10, e + 1
    return rf"\sci{{{m:.{d}f}}}{{{str(e).replace('-', chr(0x2212))}}}"


def auto(x, sig=3) -> str:
    """Plain digits for 1e-3 <= |x| < 1e6, scientific otherwise."""
    if x is None or not np.isfinite(x):
        return r"\na"
    if x == 0:
        return r"\num{0}"
    ax = abs(x)
    if 1e-3 <= ax < 1e6:
        d = max(0, sig - 1 - int(np.floor(np.log10(ax))))
        return num(x, min(d, 4))
    return sci(x, sig - 1)


def rng(pair, fmt=auto) -> str:
    lo, hi = pair
    if lo is None:
        return r"\na"
    a, b = fmt(lo), fmt(hi)
    return a if a == b else f"{a}--{b}"


def _short(h, n=16):
    return r"\na" if not h else rf"\texttt{{{h[:n]}}}\textcolor{{lightgrey}}{{\texttt{{…}}}}"


# =============================================================================== figures
def _figures(card: dict, outdir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    from . import graphite as G

    G.setup()
    d, f = card["domain"], card["forcing"]
    lo, hi = d["lower_alt_km"], d["upper_alt_km"]
    edges = np.asarray(d["edge_alt_km"])
    log = lo > 0

    # domain ruler
    fig, ax = plt.subplots(figsize=(7.3, 0.86))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.84, bottom=0.38)
    ax.hlines(0, lo, hi, color=G.CHARCOAL, lw=0.9)
    ax.vlines(edges[1:-1], -0.28, 0.28, color=G.CHARCOAL, lw=0.45)
    ax.vlines(lo, -0.62, 0.62, color=G.CHARCOAL, lw=1.6)
    ax.vlines(hi, -0.62, 0.62, color=G.CHARCOAL, lw=1.6)
    span = (hi / lo) if log else (hi - lo)
    pad = span ** 0.035 if log else 0.035 * span
    ax.axvspan(hi, hi * pad if log else hi + pad, color=G.RULE, lw=0)
    ax.text(lo, 0.95, f"{lo:,.0f} km".replace(",", " "), ha="left", va="bottom", fontproperties=G.mono(6.6),
            color=G.CHARCOAL)
    ax.text(hi, 0.95, f"{hi:,.0f} km".replace(",", " "), ha="right", va="bottom", fontproperties=G.mono(6.6),
            color=G.CHARCOAL)
    mid = np.sqrt(lo * hi) if log else 0.5 * (lo + hi)
    ax.text(mid, 0.95, f"{ubc_name(d['ubc'])} upper boundary", ha="center", va="bottom",
            fontsize=6.6, color=G.GREY)
    if f["in_run"]:
        x = np.linspace(0, 1, 60)
        xs = lo * (1.18 ** (0.25 + x)) if log else lo + (0.02 + 0.05 * x) * (hi - lo)
        ax.plot(xs, -0.85 + 0.17 * np.sin(2 * np.pi * 2 * x), color=G.CHARCOAL, lw=0.8)
        ax.text(xs[-1] * (1.02 if log else 1.0), -0.85, " flux wave at the lower boundary", fontsize=6.2,
                color=G.GREY, va="center")
    if log:
        ax.set_xscale("log")
        ax.set_xlim(lo / pad, hi * pad * pad)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        ax.set_xlim(lo - pad, hi + 2 * pad)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}".replace(",", " ")))
    ax.set_ylim(-1.45, 1.6)
    ax.set_yticks([])
    ax.grid(False)
    for s in ("left", "right", "top"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_position(("data", -1.45))
    ax.set_xlabel("altitude [km]" + (", log scale" if log else ""), fontsize=6.8, labelpad=2)
    G.tick_mono(ax, size=6.0)
    fig.savefig(outdir / "domain.pdf")
    plt.close(fig)

    # inventory panels
    names = card["run"]["species"]
    pop = card["populations"]
    tk = np.asarray(pop["time_s"]) / 1e3
    fig, axes = plt.subplots(1, len(names), figsize=(7.3, 1.55), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.23, wspace=0.18)
    for ax, nm in zip(axes[0], names):
        y = np.asarray(pop["npart"][nm], float)
        scale = 10 ** int(np.floor(np.log10(max(y.max(), 1.0)))) if y.max() >= 1e4 else 1
        ax.plot(tk, y / scale, color=G.species_color(nm), lw=1.0)
        if f["in_run"]:
            ax.axvline(f["start_time_s"] / 1e3, color=G.GREY, lw=0.7, ls=(0, (1, 1.6)))
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, tk[-1])
        unit = f" [10$^{{{int(np.log10(scale))}}}$]" if scale > 1 else ""
        ax.set_title(f"{G.species_label(nm)}  computational particles in domain{unit}", fontsize=7.2)
        ax.set_xlabel("time [ks]", fontsize=6.6, labelpad=1)
        G.tick_mono(ax, size=5.8)
    fig.savefig(outdir / "inventory.pdf")
    plt.close(fig)


# =============================================================================== LaTeX
_PREAMBLE = r"""\documentclass[9pt]{extarticle}
\usepackage[letterpaper,margin=0.6in,top=0.55in,bottom=0.55in]{geometry}
\usepackage{fontspec}
\usepackage{unicode-math}
\setmainfont{InstrumentSans}[Path=fonts/, Extension=.ttf, UprightFont=*-Regular, BoldFont=*-Bold,
  ItalicFont=*-Italic, BoldItalicFont=*-BoldItalic, Numbers=Tabular]
\setmonofont{JetBrainsMono}[Path=fonts/, Extension=.ttf, UprightFont=*-Regular, BoldFont=*-Bold,
  Scale=0.86, HyphenChar=None]
\setmathfont{FiraMath-Regular.otf}
\usepackage[table]{xcolor}
\definecolor{charcoal}{HTML}{1F1F1D}\definecolor{graphite}{HTML}{4A4A47}
\definecolor{grey}{HTML}{7A7A76}\definecolor{lightgrey}{HTML}{B9B8B3}\definecolor{rulegrey}{HTML}{E6E5E1}
\usepackage{graphicx,booktabs,tabularx,array}
\arrayrulecolor{lightgrey}
\setlength{\heavyrulewidth}{0.5pt}\setlength{\lightrulewidth}{0.35pt}
\setlength{\parindent}{0pt}\setlength{\parskip}{0pt}
\pagestyle{empty}
\linespread{1.08}
\setlength{\emergencystretch}{2em}
\newcommand{\num}[1]{{\ttfamily #1}}
\newcommand{\sci}[2]{{\ttfamily #1×10\textsuperscript{#2}}}
\newcommand{\na}{{\color{lightgrey}n/a}}
\newcommand{\sect}[1]{\par\vspace{6pt}{\scriptsize\ttfamily\color{grey}#1}\par\vspace{1pt}%
  {\color{rulegrey}\rule{\linewidth}{0.5pt}}\par\vspace{2pt}}
\newcommand{\kvhead}[1]{{\scriptsize\ttfamily\color{grey}#1}}
\newenvironment{kv}{\footnotesize\renewcommand{\arraystretch}{1.12}\begin{tabular}{@{}>{\color{graphite}}l@{\hspace{8pt}}l@{}}}{\end{tabular}}
\newcommand{\OK}{{\ttfamily\color{grey}OK}}
\newcommand{\NOTE}{{\ttfamily\color{charcoal}NOTE}}
\newcommand{\WARN}{{\ttfamily\bfseries\color{charcoal}WARN}}
\usepackage[hidelinks]{hyperref}
"""


def _status_tex(status) -> str:
    tag = {"ok": r"\OK", "note": r"\NOTE", "warn": r"\WARN"}
    items = []
    for s in status:
        det = rf" {{\color{{grey}}{tex(s['detail'])}}}" if s["detail"] else ""
        lab = tex(s["label"]).replace("·", r"$\cdot$")
        items.append(rf"\mbox{{{tag[s['level']]}\enspace {lab}{det}}}")
    return r"\hskip 1.4em plus 0.8em\relax ".join(items)


def _latex(card: dict) -> str:
    run, d, tm, f, sp, pv = card["run"], card["domain"], card["time"], card["forcing"], card["species"], card["provenance"]
    names = run["species"]
    if d["n_zenith"] == 1 and d["n_azimuthal"] == 1:
        geom = "1-D spherical"
    elif d["n_zenith"] == 1 or d["n_azimuthal"] == 1:
        geom = "2-D spherical"
    else:
        geom = "3-D spherical"
    forced = (f"forced ({tex((f['type'] or 'wave').lower())} flux)" if f["in_run"] else "unforced")
    subtitle = (" + ".join(tex_species(n) for n in names)
                + rf" \enspace·\enspace {geom} \enspace·\enspace {num(d['n_cells'])} cells"
                + rf" \enspace·\enspace {num(tm['t_end_s'])} s \enspace·\enspace {forced}")
    ctx = []
    if run["campaign"]:
        ctx.append(rf"campaign {tex_path(run['campaign'])}")
    if pv["run_config"]:
        ctx.append(rf"config {tex_path(pv['run_config'])}")
    if run["role"]:
        ctx.append(rf"role {tex(run['role'])}")

    ubc = ubc_name(d["ubc"])
    dom = [("planet radius", rf"{num(d['planet_radius_km'], 1)} km"),
           ("lower boundary", rf"{num(d['lower_alt_km'], 1)} km"),
           ("upper boundary", rf"{num(d['upper_alt_km'], 0)} km"),
           ("upper condition", tex(ubc)),
           (r"cells $r\times\theta\times\varphi$", rf"{num(d['n_radial'])} × {num(d['n_zenith'])} × {num(d['n_azimuthal'])}"),
           (r"$\Delta r$ min / max", rf"{auto(d['dr_min_km'])} / {auto(d['dr_max_km'])} km")]
    if d["n_zenith"] > 1:
        dom.append(("zenith span", rf"{num(d['zenith_deg'][0], 1)}--{num(d['zenith_deg'][1], 1)}°"))
    if d["n_azimuthal"] > 1:
        dom.append(("azimuth span", rf"{num(d['azimuth_deg'][0], 1)}--{num(d['azimuth_deg'][1], 1)}°"))
    tim = [(r"time step $\Delta t$", rf"{auto(tm['dt_s'], 4)} s"),
           ("steps", num(tm["n_steps"])),
           ("simulated time", rf"{num(tm['t_end_s'])} s"),
           ("sampled every", rf"{num(tm['samp_steps'])} step(s)"),
           ("averaged over", rf"{num(tm['avg_steps'])} steps = {auto(tm['avg_window_s'])} s"),
           ("outputs", num(tm["n_outputs"]))]
    if f["omega_rad_s"] > 0 and f["amplitude"] > 0:
        endnote = r"\textsuperscript{\color{grey}*}" if f["end_beyond_run"] else ""
        frc = [("wave", rf"{tex(f['type'] or 'n/a')} flux, $\varepsilon$ ={num(f['amplitude'], 2)}"),
               (r"$\omega$ / period", rf"{sci(f['omega_rad_s'], 1)} rad s\textsuperscript{{−1}} / {num(f['period_s'], 1)} s"),
               ("start", rf"step {num(f['start_step'])} = {num(f['start_time_s'])} s"),
               ("end", rf"step {num(f['end_step'])} = {num(f['end_time_s'])} s{endnote}"),
               ("periods in run", num(f["periods_in_run"], 2) if f["in_run"] else r"0 {\color{grey}\scriptsize (starts after run end)}"),
               (r"horiz. $\lambda$ / width", rf"{num(f['horizontal_wavelength_deg'], 1)}° / {num(f['injection_width_cells'])} cell(s)")]
    else:
        frc = [("wave", "none configured")]

    def kv(rows):
        return "\\begin{kv}\n" + "\n".join(rf"{a} & {b} \\" for a, b in rows) + "\n\\end{kv}"

    setup_rows = []
    for s in sp:
        setup_rows.append(" & ".join([
            tex_species(s["name"]), num(s["mass_amu"], 3), sci(s["weight"]), rng(s["n0_m3"], sci), rng(s["T0_K"]),
            rng(s["speed_m_s"]), rng(s["flux_m2_s"], sci), rng(s["flow_rate_s"], sci),
            rng(s["jeans_lambda0"]), rng(s["jeans_rate0_s"], sci)]) + r" \\")
    pop_rows = []
    for s in sp:
        pop_rows.append(" & ".join([
            tex_species(s["name"]), num(s["npart_final"]), num(s["npart_peak"]), sci(s["nphys_final"]),
            rng(s["npart_bottom_cell_tail"], lambda x: num(x, 1)), rng(s["npart_top_cell_tail"], lambda x: num(x, 1)),
            num(s["npart_injected"]), num(s["npart_returned"]), num(s["npart_escaped"])]) + r" \\")
    coll = r"\hspace{1.4em}".join(
        rf"\mbox{{{tex_species(c['species_1'])}--{tex_species(c['species_2'])} {{\color{{grey}}$\mu$}} {num(c['reduced_mass_amu'], 3)} amu"
        rf" {{\color{{grey}}\scriptsize type {c['type']}}}}}" for c in card["collisions"]) or r"\na"

    sha = pv["sha256"]
    prov_l = [("commit", (rf"\texttt{{{tex(pv['commit'][:12])}}}" if pv["commit"] else r"\na")
               + (rf" {{\color{{grey}}on}} {tex_path(pv['branch'])}" if pv["branch"] else "")),
              ("host / MPI ranks", rf"{tex(pv['host'] or 'n/a')} / {num(pv['mpi_ranks']) if pv['mpi_ranks'] else r'\na'}"),
              ("started", rf"\texttt{{{tex(pv['started'])}}}" if pv["started"] else r"\na"),
              ("finished", rf"\texttt{{{tex(pv['finished'])}}}" if pv["finished"] else r"\na"),
              ("wall time", rf"{num(pv['wall_s'])} s" if pv["wall_s"] is not None else r"\na")]
    prov_r = [("builder", (rf"\texttt{{{tex(pv['builder_version'])}}}" if pv["builder_version"] else r"\na")
               + (r" {\color{grey}strict}" if pv["strict_mode"] else "")),
              ("sha256 run file", _short(sha["run_file"])),
              ("sha256 Input/", _short(sha["input"]) + rf" {{\color{{grey}}({pv['n_input_files']} files)}}"),
              ("sha256 run\\_meta", _short(sha["run_meta"])),
              ("sha256 DSMC binary", _short(sha["executable"]))]
    tail_pct = 100 * card["populations"]["tail_fraction"]
    forcing_note = (r"\par\vspace{1pt}{\scriptsize\color{grey}* configured end lies after the last simulated step}"
                    if f["omega_rad_s"] > 0 and f["amplitude"] > 0 and f["end_beyond_run"] else "")
    gen = card["generated"].replace("+00:00", "Z")

    body = rf"""
\begin{{document}}\color{{charcoal}}
{{\scriptsize\ttfamily\color{{grey}}SIMULATION CARD \enspace·\enspace MARSATMOS DSMC\hfill card {CARD_VERSION} \enspace·\enspace dsmclib {tex(pv['dsmclib_version'])} \enspace·\enspace {tex(gen)}}}\par
\vspace{{7pt}}
{{\fontsize{{20}}{{23}}\selectfont\bfseries {tex(run['alias'])}}}\par\vspace{{2pt}}
{{\fontsize{{10.5}}{{13}}\selectfont\color{{graphite}} {subtitle}}}\par\vspace{{2pt}}
{{\small\color{{grey}} {r' \enspace·\enspace '.join(ctx)}}}\par
\vspace{{6pt}}{{\color{{rulegrey}}\rule{{\linewidth}}{{0.6pt}}}}\par\vspace{{4pt}}
{{\small\raggedright {_status_tex(card['status'])}\par}}

\sect{{DOMAIN}}
\includegraphics[width=\linewidth]{{domain.pdf}}\par\vspace{{3pt}}
\begin{{minipage}}[t]{{0.315\linewidth}}\kvhead{{GRID}}\par\vspace{{1pt}}{kv(dom)}\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.315\linewidth}}\kvhead{{TIME}}\par\vspace{{1pt}}{kv(tim)}\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.33\linewidth}}\kvhead{{FORCING}}\par\vspace{{1pt}}{kv(frc)}{forcing_note}\end{{minipage}}\par\vspace{{4pt}}

\sect{{SPECIES AT THE LOWER BOUNDARY \enspace(Info/Surface\_Cells)}}
{{\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}l rrrrrrrrr@{{}}}}
Species & $m$ [amu] & weight $W$ & $n_0$ [m$^{{-3}}$] & $T_0$ [K] & $\bar c$ [m s$^{{-1}}$] & flux [m$^{{-2}}$ s$^{{-1}}$] & flow [s$^{{-1}}$] & $\lambda_0$ & Jeans [s$^{{-1}}$] \\
\midrule
{chr(10).join(setup_rows)}
\bottomrule
\end{{tabular*}}}}\par\vspace{{4pt}}
{{\footnotesize\color{{graphite}}Collisions\quad}}{{\footnotesize {coll}}}\par

\sect{{POPULATIONS \enspace(computational particles)}}
{{\footnotesize\setlength{{\tabcolsep}}{{3pt}}
\begin{{tabular*}}{{\linewidth}}{{@{{\extracolsep{{\fill}}}}l rrrrrrrr@{{}}}}
Species & final & peak & physical, final & bottom cell & top cell & injected & returned & escaped \\
\midrule
{chr(10).join(pop_rows)}
\bottomrule
\end{{tabular*}}}}\par\vspace{{2pt}}
{{\scriptsize\color{{grey}}Bottom/top cell: mean computational particles per cell over the last {num(tail_pct, 0)}\,\% of outputs (range over angular cells). Injected, returned and escaped: totals over the run from the boundary rates.}}\par\vspace{{2pt}}
\includegraphics[width=\linewidth]{{inventory.pdf}}\par

\sect{{PROVENANCE}}
\begin{{minipage}}[t]{{0.49\linewidth}}{kv(prov_l)}\end{{minipage}}\hfill
\begin{{minipage}}[t]{{0.49\linewidth}}{kv(prov_r)}\end{{minipage}}\par\vspace{{4pt}}
{{\scriptsize\color{{graphite}}
\begin{{tabularx}}{{\linewidth}}{{@{{}}>{{\color{{grey}}}}l@{{\hspace{{8pt}}}}X@{{}}}}
run folder & {tex_path(pv['paths']['run_dir'])} \\
Run file & {tex_path(pv['paths']['run_file'])} \\
Run-file history & {tex(pv['run_file_history'] or 'n/a')} \\
\end{{tabularx}}}}
\vfill
{{\scriptsize\color{{lightgrey}}Generated by \texttt{{dsmclib.card}} from the run file and the run folder; no value is typed by hand. JSON twin: \texttt{{{tex(run['alias'])}\_card.json}}.}}
\end{{document}}
"""
    return _PREAMBLE + body


def _page_count(pdf: Path) -> int:
    data = pdf.read_bytes()
    counts = [int(m) for m in re.findall(rb"/Type\s*/Pages\b[^>]*?/Count\s+(\d+)", data)]
    counts += [int(m) for m in re.findall(rb"/Count\s+(\d+)[^>]*?/Type\s*/Pages\b", data)]
    if counts:
        return max(counts)
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        out = subprocess.run([pdfinfo, str(pdf)], capture_output=True, text=True).stdout
        m = re.search(r"Pages:\s+(\d+)", out)
        if m:
            return int(m.group(1))
    return -1


def render(card: dict, out_pdf: Path) -> Path:
    """Render the card to `out_pdf` (one US Letter page, enforced)."""
    from . import graphite as G

    tectonic = shutil.which("tectonic")
    if not tectonic:
        raise RuntimeError("tectonic not found on PATH (brew install tectonic)")
    with tempfile.TemporaryDirectory(prefix="simcard_") as td:
        td = Path(td)
        (td / "fonts").mkdir()
        for f in G.FONTS.glob("*.ttf"):
            shutil.copy(f, td / "fonts" / f.name)
        _figures(card, td)
        (td / "card.tex").write_text(_latex(card), encoding="utf-8")
        proc = subprocess.run([tectonic, "--keep-logs", "-o", str(td), "card.tex"], cwd=td,
                              capture_output=True, text=True)
        pdf = td / "card.pdf"
        if proc.returncode != 0 or not pdf.is_file():
            raise RuntimeError(f"tectonic failed for {card['run']['alias']}:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
        pages = _page_count(pdf)
        if pages != 1:
            raise RuntimeError(f"card for {card['run']['alias']} has {pages} pages; the card must fit one page")
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(pdf, out_pdf)
    return out_pdf


# =============================================================================== outputs
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _index_row(card: dict, pdf: Path) -> dict:
    d, tm, f, pv = card["domain"], card["time"], card["forcing"], card["provenance"]
    return {"alias": card["run"]["alias"], "campaign": card["run"]["campaign"] or "",
            "species": "+".join(card["run"]["species"]), "n_radial": d["n_radial"], "n_zenith": d["n_zenith"],
            "n_azimuthal": d["n_azimuthal"], "lower_alt_km": round(d["lower_alt_km"], 3),
            "upper_alt_km": round(d["upper_alt_km"], 3), "ubc": d["ubc"], "dt_s": tm["dt_s"], "n_steps": tm["n_steps"],
            "t_end_s": round(tm["t_end_s"], 3), "n_outputs": tm["n_outputs"], "forced": f["in_run"],
            "periods_in_run": round(f["periods_in_run"] or 0.0, 3),
            "warnings": sum(s["level"] == "warn" for s in card["status"]),
            "commit": (pv["commit"] or "")[:12], "builder": pv["builder_version"] or "",
            "run_file_sha256": pv["sha256"]["run_file"] or "", "generated": card["generated"], "card_pdf": str(pdf)}


def write_index(folder: Path = CENTRAL) -> Path:
    """Rebuild cards.csv from every *_card.json in the central folder."""
    rows = []
    for j in sorted(folder.glob(f"*{CARD_SUFFIX}.json")):
        card = json.loads(j.read_text())
        rows.append(_index_row(card, j.with_suffix(".pdf")))
    out = folder / "cards.csv"
    if rows:
        with open(out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return out


def make_card(target, overwrite: bool = False, alias: Optional[str] = None, central: bool = True) -> dict:
    """Extract, render and write a run's card next to its run file and (central=True) in outputs/cards/.

    Existing cards are kept unless overwrite=True. Returns {'status', 'pdf', 'json', 'alias'}."""
    rf, alias0 = find_run_file(target)
    results = rf.parent
    alias = alias or alias0
    if alias is None:
        reg = _registry_match(results)
        alias = reg.alias if reg else rf.name[: -len(RUN_SUFFIX)]
    pdf, js = results / f"{alias}{CARD_SUFFIX}.pdf", results / f"{alias}{CARD_SUFFIX}.json"
    if pdf.exists() and js.exists() and not overwrite:
        status = "exists, kept"
    else:
        card = extract(rf, alias=alias)
        render(card, pdf)
        js.write_text(json.dumps(card, indent=1, default=_json_default))
        status = "written"
    if central:
        CENTRAL.mkdir(parents=True, exist_ok=True)
        for src in (pdf, js):
            dst = CENTRAL / src.name
            if overwrite or not dst.exists() or status == "written":
                shutil.copy(src, dst)
        write_index()
    return {"status": status, "pdf": str(pdf), "json": str(js), "alias": alias}


def _targets(args) -> list:
    from . import registry

    if args.run_file:
        return [(Path(p), None) for p in args.run_file]
    if args.all:
        runs = registry.all_runs()
    elif args.campaign:
        runs = registry.campaign(args.campaign)
    else:
        runs = [registry.run(a) for a in args.aliases]
    out = []
    for r in runs:
        if (r.path / f"{r.alias}{RUN_SUFFIX}").is_file():
            out.append((r.alias, r.alias))
        elif args.aliases and r.alias in args.aliases:
            out.append((r.alias, r.alias))  # explicit request: let find_run_file report what is missing
        else:
            print(f"{r.alias}: no run file, skipped (build it with python -m dsmclib.runfile)", flush=True)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("aliases", nargs="*", help="registry aliases")
    parser.add_argument("--campaign", help="every run of this campaign that has a run file")
    parser.add_argument("--all", action="store_true", help="every registered run that has a run file")
    parser.add_argument("--run-file", nargs="+", help="run file path(s), registered or not")
    parser.add_argument("--overwrite", action="store_true", help="re-render existing cards")
    parser.add_argument("--no-central", action="store_true", help="do not copy to outputs/cards/")
    args = parser.parse_args(argv)
    if not (args.aliases or args.campaign or args.all or args.run_file):
        parser.error("give aliases, --campaign, --all or --run-file")
    failed = 0
    for target, alias in _targets(args):
        try:
            res = make_card(target, overwrite=args.overwrite, alias=alias, central=not args.no_central)
            print(f"{res['alias']}: card {res['status']} -> {res['pdf']}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"{target}: card FAILED: {exc}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
