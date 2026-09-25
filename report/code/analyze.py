"""Analyze data/run.csv: noise statistics, random-number generation, figures, report.

  python3 analyze.py

Random numbers: the least-significant bit (LSB) of each channel of each sample is
a raw random bit (the noise is several counts wide, so the last digit is
unpredictable). A von Neumann extractor removes bias: bits are read in pairs,
01 -> 0, 10 -> 1, 00 and 11 are discarded. The clean bits are tested, saved to
data/random_bits.txt and data/random_bytes.bin, and turned into dice rolls.

Writes report/results.tex, report/tests_table.tex, report/figures/*.png, copies
the code into report/code/ and zips the report folder as report_overleaf.zip.
"""
import csv
import math
import os
import shutil

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sensor import saturation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
REPORT = os.path.join(HERE, "report")
FIG = os.path.join(REPORT, "figures")
BLUE, ORANGE, INK, INK2, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
ALPHA = 0.01

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def fmt(x, nd=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:,.{nd}f}".replace(",", "{,}")


def von_neumann(bits):
    pairs = bits[: len(bits) // 2 * 2].reshape(-1, 2)
    return pairs[pairs[:, 0] != pairs[:, 1], 0]


def randomness_tests(bits):
    """Four quick tests in the style of NIST SP 800-22; each returns a p-value."""
    n = len(bits)
    if n < 100:
        return {k: float("nan") for k in ("monobit", "runs", "serial", "lag1")}
    s = 2 * bits.astype(float) - 1
    out = {"monobit": math.erfc(abs(s.sum()) / math.sqrt(2 * n))}
    pi = bits.mean()
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        out["runs"] = 0.0
    else:
        v = 1 + int(np.sum(bits[1:] != bits[:-1]))
        out["runs"] = math.erfc(abs(v - 2 * n * pi * (1 - pi)) / (2 * math.sqrt(2 * n) * pi * (1 - pi)))
    pairs = bits[: n // 2 * 2].reshape(-1, 2)
    counts = np.bincount(pairs[:, 0] * 2 + pairs[:, 1], minlength=4)
    m = len(pairs) / 4
    x = float(np.sum((counts - m) ** 2 / m))
    out["serial"] = math.erfc(math.sqrt(x / 2)) + math.sqrt(2 * x / math.pi) * math.exp(-x / 2)
    r = float(np.mean(s[1:] * s[:-1]))
    out["lag1"] = math.erfc(abs(r) * math.sqrt(n - 1) / math.sqrt(2))
    return out


def tests_table(raw_t, vn_t, n_raw, n_vn):
    names = {"monobit": "Frequency (monobit)", "runs": "Runs", "serial": r"Serial (2-bit, $\chi^2$)",
             "lag1": "Lag-1 autocorrelation"}

    def cell(v):
        if not np.isfinite(v):
            return "n/a"
        return f"{v:.3f} (" + ("pass" if v >= ALPHA else r"\textbf{fail}") + ")"

    lines = [r"\begin{tabular}{lcc}", r"\toprule",
             rf"Test & Raw LSBs ({fmt(n_raw, 0)} bits) & Debiased ({fmt(n_vn, 0)} bits) \\", r"\midrule"]
    lines += [f"{names[k]} & {cell(raw_t[k])} & {cell(vn_t[k])} \\\\" for k in names]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def plots(t, c0, d0, vn):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.4, 2.6), gridspec_kw={"width_ratios": [1.6, 1]})
    k = t <= 60
    a1.plot(t[k], c0[k], color=BLUE, lw=1)
    a1.set_xlabel("Time (s)")
    a1.set_ylabel("ch0 (counts)")
    lo, hi = np.percentile(d0, [0.5, 99.5])
    bins = np.arange(math.floor(lo) - 0.5, math.ceil(hi) + 1.5, max(1, round((hi - lo) / 40)))
    a2.hist(d0, bins=bins, density=True, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.5)
    sd = d0.std()
    xs = np.linspace(bins[0], bins[-1], 300)
    a2.plot(xs, np.exp(-xs ** 2 / (2 * sd ** 2)) / (sd * math.sqrt(2 * math.pi)), color=ORANGE, lw=2)
    a2.set_xlabel(r"Change $\Delta x$ (counts)")
    a2.set_ylabel("Density")
    a2.grid(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "noise.png"))
    plt.close(fig)

    side = int(math.isqrt(min(len(vn), 4096)))
    if side >= 8:
        fig, ax = plt.subplots(figsize=(2.4, 2.4))
        ax.imshow(vn[: side * side].reshape(side, side), cmap="gray_r", interpolation="nearest")
        ax.set_axis_off()
        fig.savefig(os.path.join(FIG, "bits.png"))
        plt.close(fig)


def main():
    path = os.path.join(DATA, "run.csv")
    if not os.path.exists(path):
        raise SystemExit("No data/run.csv yet: run  python3 collect.py run  first.")
    with open(path) as f:
        meta = dict(kv.split("=", 1) for kv in f.readline().lstrip("#").split())
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t_s"]) for r in rows])
    c0 = np.array([int(r["ch0"]) for r in rows])
    c1 = np.array([int(r["ch1"]) for r in rows])
    simulated = meta.get("simulated") == "1"
    sat = bool(c0.max() >= 0.95 * saturation(int(meta["atime_ms"])))

    # noise: sample-to-sample changes (differencing removes slow drift of the phone light)
    d0, d1 = np.diff(c0).astype(float), np.diff(c1).astype(float)
    sigma0, sigma1 = d0.std() / math.sqrt(2), d1.std() / math.sqrt(2)
    corr = float(np.corrcoef(d0, d1)[0, 1]) if d0.std() > 0 and d1.std() > 0 else float("nan")

    # random numbers
    raw = np.empty(2 * len(c0), dtype=np.uint8)
    raw[0::2], raw[1::2] = c0 & 1, c1 & 1
    vn = von_neumann(raw)
    raw_t, vn_t = randomness_tests(raw), randomness_tests(vn)
    fails = [k for k, v in vn_t.items() if not (v >= ALPHA)]
    with open(os.path.join(DATA, "random_bits.txt"), "w") as f:
        f.write("".join(map(str, vn)) + "\n")
    nbytes = len(vn) // 8
    np.packbits(vn[: nbytes * 8]).tofile(os.path.join(DATA, "random_bytes.bin"))
    # dice: 3 bits -> 0..7, keep 0..5 (rejection sampling keeps every face equally likely)
    tri = vn[: len(vn) // 3 * 3].reshape(-1, 3) @ np.array([4, 2, 1])
    dice = (tri[tri < 6] + 1)[:20]

    if len(vn) < 100:
        verdict = r"Too few debiased bits for the randomness tests; record a longer run."
    elif not fails:
        verdict = (rf"The debiased stream passes all four tests at $\alpha={ALPHA}$: no detectable "
                   rf"bias or correlation in {fmt(len(vn), 0)} bits.")
    else:
        verdict = (rf"The debiased stream fails {len(fails)} of 4 tests at $\alpha={ALPHA}$ "
                   rf"({', '.join(fails)}), so it is not a clean random source at this light level.")
    if not np.isfinite(corr):
        flicker = "n/a"
    elif abs(corr) < 0.2:
        flicker = (rf"The noise on the two photodiodes is nearly uncorrelated ($r={fmt(corr)}$), so it is "
                   r"mostly independent per-diode noise (photon shot noise plus read noise) rather than "
                   r"flicker of the phone light, which would move both channels together.")
    else:
        flicker = (rf"The noise on the two photodiodes is correlated ($r={fmt(corr)}$): part of it is "
                   r"flicker of the phone light, which moves both channels together.")

    os.makedirs(FIG, exist_ok=True)
    plots(t, c0, d0, vn)
    with open(os.path.join(REPORT, "tests_table.tex"), "w") as f:
        f.write(tests_table(raw_t, vn_t, len(raw), len(vn)))
    m = {"isSimulated": int(simulated), "runGain": meta["gain"], "runAtime": meta["atime_ms"],
         "runSamples": fmt(len(c0), 0), "runMinutes": fmt((t[-1] - t[0]) / 60, 1),
         "runMean": fmt(c0.mean(), 0), "runMeanIR": fmt(c1.mean(), 0), "runSigma": fmt(sigma0),
         "runSigmaIR": fmt(sigma1), "runCorr": fmt(corr), "runRawBits": fmt(len(raw), 0),
         "runBits": fmt(len(vn), 0), "runBytes": fmt(nbytes, 0),
         "runYield": fmt(100 * len(vn) / max(len(raw), 1), 0),
         "runDice": " ".join(map(str, dice)) if len(dice) else "n/a",
         "runFlicker": flicker, "runVerdict": verdict + (r" \textbf{Warning: the run saturated.}" if sat else "")}
    with open(os.path.join(REPORT, "results.tex"), "w") as f:
        f.write("% generated by analyze.py - do not edit by hand\n")
        f.write("".join(f"\\renewcommand{{\\{k}}}{{{v}}}\n" for k, v in m.items()))

    code_dir = os.path.join(REPORT, "code")
    os.makedirs(code_dir, exist_ok=True)
    for fn in ("sensor.py", "collect.py", "analyze.py", "sonify.py", "waves.py", "sonic_pi.rb"):
        shutil.copy(os.path.join(HERE, fn), code_dir)
    shutil.make_archive(os.path.join(HERE, "report_overleaf"), "zip", REPORT)

    print(f"\n{'SIMULATED ' if simulated else ''}run: {len(c0)} samples over {(t[-1] - t[0]) / 60:.1f} min")
    print(f"  ch0 mean {c0.mean():.0f} counts, noise {sigma0:.2f} counts   ch1 mean {c1.mean():.0f}, noise {sigma1:.2f}")
    print(f"  ch0/ch1 noise correlation r = {corr:.2f}  ({'mostly per-diode noise' if abs(corr) < 0.2 else 'light flicker present'})")
    if sat:
        print("  WARNING: saturated. Move the phone farther and record again.")
    if sigma0 < 1.5:
        print("  WARNING: noise under ~1.5 counts; LSBs are weakly random. Move the phone closer.")
    print(f"\nRandom numbers: {len(raw)} raw bits -> {len(vn)} clean bits ({nbytes} bytes)")
    for k, v in vn_t.items():
        print(f"  {k:>8}: p = {v:.3f}  {'pass' if v >= ALPHA else 'FAIL'}")
    print(f"  first 64 bits: {''.join(map(str, vn[:64]))}")
    print(f"  dice rolls   : {' '.join(map(str, dice))}")
    print("\nSaved data/random_bits.txt, data/random_bytes.bin; report updated -> report_overleaf.zip")
    if simulated:
        print("NOTE: data is SIMULATED; the report will carry a red 'simulated' banner.")


if __name__ == "__main__":
    main()
