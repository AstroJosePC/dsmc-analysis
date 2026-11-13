"""Check that every macro used by main.tex (and tables/*.tex) is defined.

A macro counts as defined if it is (a) defined in numbers.tex, (b) defined by
\\newcommand/\\def in main.tex, or (c) known to the LaTeX preamble of main.tex
(packages, kernel).  (c) is tested with a tectonic probe that compiles the
preamble of main.tex *without* numbers.tex and asks \\ifcsname for every
candidate.  Exits 1 if anything used is undefined; also lists numbers.tex
macros that main.tex never uses (informational).

Usage (from the report folder):
    python -B scripts/check_macros.py            # tectonic probe (default)
    python -B scripts/check_macros.py --static   # skip probe; only check
                                                 # number-style macros
"""
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CS = re.compile(r"\\([A-Za-z]+)")
DEF = re.compile(r"\\(?:re)?newcommand\*?\{?\\([A-Za-z]+)\}?|\\def\\([A-Za-z]+)")


def strip_comments(text):
    return re.sub(r"(?<!\\)%.*", "", text)


def used_macros():
    src = strip_comments((ROOT / "main.tex").read_text())
    body = src.split(r"\begin{document}", 1)[1]
    used = set(CS.findall(body))
    for t in sorted((ROOT / "tables").glob("*.tex")):
        used |= set(CS.findall(strip_comments(t.read_text())))
    return used, src


def defined_in(text):
    return {a or b for a, b in DEF.findall(text)}


def probe_undefined(preamble, names):
    """Return the subset of names that the preamble (w/o numbers.tex) leaves undefined."""
    scratch = ROOT / "build" / "check_macros"
    scratch.mkdir(parents=True, exist_ok=True)
    pre = preamble.replace(r"\input{numbers.tex}", "")
    # fonts/ and figures/ are relative to ROOT; run tectonic from ROOT.
    lines = [pre, r"\begin{document}"]
    for n in sorted(names):
        lines.append(r"\ifcsname %s\endcsname\else\typeout{UNDEF:%s}\fi" % (n, n))
    lines.append(r"\typeout{PROBE-DONE}x\end{document}")
    # the probe must sit in ROOT so that fontspec's Path=fonts/ resolves
    probe = ROOT / "_check_macros_probe.tex"
    probe.write_text("\n".join(lines))
    try:
        out = subprocess.run(
            ["tectonic", "--print", "-o", str(scratch), probe.name],
            cwd=ROOT, capture_output=True, text=True)
    finally:
        probe.unlink(missing_ok=True)
    log = out.stdout + out.stderr
    if "PROBE-DONE" not in log:
        sys.exit("probe compile failed:\n" + log[-3000:])
    return set(re.findall(r"UNDEF:([A-Za-z]+)", log))


def main():
    static = "--static" in sys.argv
    used, src = used_macros()
    numbers = defined_in((ROOT / "numbers.tex").read_text())
    local = defined_in(src)
    candidates = used - numbers - local
    if static or shutil.which("tectonic") is None:
        # only number-style macros (lowerCamelCase containing an upper-case letter)
        missing = {n for n in candidates
                   if re.match(r"^[a-z]+[A-Z]", n) and n not in {"textsubscript", "textsuperscript"}}
    else:
        preamble = src.split(r"\begin{document}", 1)[0]
        missing = probe_undefined(preamble, candidates)
    unused = sorted(numbers - used)
    print(f"used macros: {len(used)}; numbers.tex defines {len(numbers)}; "
          f"main.tex defines {len(local)}")
    if unused:
        print(f"numbers.tex macros not used by main.tex ({len(unused)}): " + ", ".join(unused))
    if missing:
        print("UNDEFINED macros used in main.tex/tables: " + ", ".join(sorted(missing)))
        sys.exit(1)
    print("OK: every macro used is defined")


if __name__ == "__main__":
    main()
