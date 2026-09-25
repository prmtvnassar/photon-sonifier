"""Math sound waves from your recorded sensor noise, played on a USB speaker.

  python3 waves.py pluck            Karplus-Strong strings, excited by the sensor's own noise
  python3 waves.py fm               FM bells; the noise sets the modulation index (brightness)
  add --device plughw:1,0 to pick the USB speaker (find it with: aplay -l)

Reads data/run.csv, writes waves_<mode>.wav, then plays it with aplay.
"""
import argparse
import csv
import os
import subprocess
import wave

import numpy as np

from sonify import NoteMaker

HERE = os.path.dirname(os.path.abspath(__file__))
SR = 22050
PHI = (1 + 5 ** 0.5) / 2  # golden ratio: inharmonic, bell-like FM


def load():
    with open(os.path.join(HERE, "data", "run.csv")) as f:
        f.readline()
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t_s"]) for r in rows])
    c0 = np.array([int(r["ch0"]) for r in rows])
    c1 = np.array([int(r["ch1"]) for r in rows])
    return t, c0, c1


def notes(t, c0, c1, seconds):
    maker, ev = NoteMaker(), []
    for i in range(len(t)):
        if t[i] > seconds:
            break
        ev += [(t[i], i, n, d) for n, d in maker.feed(int(c0[i]), int(c1[i]))]
    return ev


def pluck(f0, dur, excitation, decay=0.996):
    """Karplus-Strong: y[n] = decay * (y[n-N] + y[n-N-1]) / 2, delay line seeded with real noise."""
    N = max(int(SR / f0), 2)
    buf = np.resize(excitation, N).astype(float)
    out, last = [], 0.0
    for _ in range(int(dur * SR / N) + 1):
        out.append(buf)
        buf = decay * 0.5 * (buf + np.concatenate(([last], buf[:-1])))
        last = out[-1][-1]
    return np.concatenate(out)[: int(dur * SR)]


def fm(f0, dur, index):
    """y = sin(2 pi f t + I(t) sin(2 pi phi f t)); sidebands follow Bessel functions J_k(I)."""
    tt = np.arange(int(dur * SR)) / SR
    env = np.minimum(tt / 0.005, 1) * np.exp(-tt / (0.35 * dur))
    return env * np.sin(2 * np.pi * f0 * tt + index * env * np.sin(2 * np.pi * PHI * f0 * tt))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["pluck", "fm"])
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--device", help="ALSA device for aplay, e.g. plughw:1,0")
    p.add_argument("--no-play", action="store_true")
    a = p.parse_args()

    t, c0, c1 = load()
    dx = np.diff(c0).astype(float)
    z = (dx - dx.mean()) / (dx.std() + 1e-9)  # standardized sensor noise
    local = np.convolve(np.abs(z), np.ones(25) / 25, mode="same")  # rolling noise strength
    out = np.zeros(int((min(a.seconds, t[-1]) + 4) * SR))
    for ts, i, n, d in notes(t, c0, c1, a.seconds):
        f0 = 440 * 2 ** ((n - 69) / 12)
        if a.mode == "pluck":
            seg = pluck(f0, d + 1.5, np.resize(z[i:i + 400], 400) if i + 1 < len(z) else z)
        else:
            seg = fm(f0, d + 1.0, index=4 * local[min(i, len(local) - 1)])
        s = int(ts * SR)
        out[s:s + len(seg)] += 0.3 * seg[: len(out) - s]
    out = (out / (np.abs(out).max() + 1e-9) * 0.9 * 32767).astype("<i2")
    path = os.path.join(HERE, f"waves_{a.mode}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(SR)
        w.writeframes(out.tobytes())
    print(f"Wrote {os.path.basename(path)} ({len(out) / SR:.0f} s)")
    if not a.no_play:
        subprocess.run(["aplay"] + (["-D", a.device] if a.device else []) + [path])


if __name__ == "__main__":
    main()
