"""Turn sensor noise into music.

  python3 sonify.py                       live: sensor -> OSC -> Sonic Pi on this Pi
  python3 sonify.py --host 192.168.1.20   live, Sonic Pi running on your laptop
  python3 sonify.py --wav demo.wav        offline: render data/run.csv to a WAV file
  add --simulate to run live without hardware (rehearsal only)

Mapping: the least-significant bit of each channel is a raw random bit; pairs of
raw bits go through a von Neumann extractor (01->0, 10->1, 00/11 dropped) to
remove bias. Every 4 clean bits make one note: 3 bits pick one of 8 scale notes,
1 bit picks short or long. The light level (ch0) drives the pad's filter, so
covering the pinhole audibly darkens the sound.
"""
import argparse
import csv
import math
import os
import socket
import struct
import wave

import numpy as np

from sensor import open_sensor, saturation

HERE = os.path.dirname(os.path.abspath(__file__))
SCALE = [57, 60, 62, 64, 67, 69, 72, 74]  # A minor pentatonic, MIDI numbers (8 notes = 3 bits)


class NoteMaker:
    def __init__(self):
        self.pending, self.bits = None, []

    def feed(self, ch0, ch1):
        """Feed one sample; returns a list of (midi_note, duration_s)."""
        for raw in (ch0 & 1, ch1 & 1):
            if self.pending is None:
                self.pending = raw
            else:
                if raw != self.pending:
                    self.bits.append(self.pending)
                self.pending = None
        notes = []
        while len(self.bits) >= 4:
            b0, b1, b2, b3 = self.bits[:4]
            del self.bits[:4]
            notes.append((SCALE[b0 << 2 | b1 << 1 | b2], 1.6 if b3 else 0.5))
        return notes


def light_level(ch0, sat):
    """0..1, logarithmic in counts."""
    return min(max(math.log10(max(ch0, 1)) / math.log10(sat), 0.0), 1.0)


def osc_message(address, *args):
    """Encode a minimal OSC 1.0 message (int32 / float32 arguments)."""
    def pad(b):
        return b + b"\0" * (4 - len(b) % 4)
    tags, payload = ",", b""
    for a in args:
        if isinstance(a, int):
            tags, payload = tags + "i", payload + struct.pack(">i", a)
        else:
            tags, payload = tags + "f", payload + struct.pack(">f", float(a))
    return pad(address.encode()) + pad(tags.encode()) + payload


def live(args):
    sensor = open_sensor(args.simulate, args.gain, 100, realtime=True)
    if args.simulate:
        sensor.set_level(15000)
    sat = saturation(sensor.atime_ms)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dest = (args.host, args.port)
    maker, i = NoteMaker(), 0
    print(f"Sending OSC to {args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        while True:
            _, c0, c1 = sensor.read()
            lvl = light_level(c0, sat)
            if i % 5 == 0:
                sock.sendto(osc_message("/photon/level", lvl), dest)
            for note, dur in maker.feed(c0, c1):
                sock.sendto(osc_message("/photon/note", note, dur, 0.3 + 0.5 * lvl), dest)
                print(f"  ch0 {c0:6d}  note {note}  {'long ' if dur > 1 else 'short'}")
            i += 1
    except KeyboardInterrupt:
        print()
    finally:
        sensor.close()


def render_wav(args):
    path = os.path.join(HERE, "data", "run.csv")
    with open(path) as f:
        meta = dict(kv.split("=", 1) for kv in f.readline().lstrip("#").split())
        rows = list(csv.DictReader(f))
    sat = saturation(int(meta["atime_ms"]))
    maker, events, levels = NoteMaker(), [], []
    for r in rows:
        t = float(r["t_s"])
        if t > args.seconds:
            break
        levels.append((t, light_level(int(r["ch0"]), sat)))
        events += [(t, n, d) for n, d in maker.feed(int(r["ch0"]), int(r["ch1"]))]
    sr = 22050
    total = int((min(args.seconds, levels[-1][0]) + 3) * sr)
    out = np.zeros(total)
    for t, note, dur in events:
        f0 = 440 * 2 ** ((note - 69) / 12)
        n = int((dur + 1.0) * sr)
        tt = np.arange(n) / sr
        env = np.minimum(tt / 0.005, 1) * np.exp(-tt / (0.45 * dur))
        tone = np.sin(2 * np.pi * f0 * tt) + 0.3 * np.sin(4 * np.pi * f0 * tt) + 0.1 * np.sin(6 * np.pi * f0 * tt)
        s = int(t * sr)
        seg = out[s:s + n]
        seg += 0.25 * env[: len(seg)] * tone[: len(seg)]
    tt = np.arange(total) / sr
    lvl = np.interp(tt, [x[0] for x in levels], [x[1] for x in levels])
    out += 0.06 * (0.3 + lvl) * (np.sin(2 * np.pi * 110 * tt) + 0.5 * np.sin(2 * np.pi * 164.8 * tt))
    out = (out / max(np.abs(out).max(), 1e-9) * 0.9 * 32767).astype("<i2")
    with wave.open(args.wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(out.tobytes())
    tag = "  (SIMULATED data)" if meta.get("simulated") == "1" else ""
    print(f"Wrote {args.wav}: {len(events)} notes over {total / sr:.0f} s{tag}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="machine running Sonic Pi")
    p.add_argument("--port", type=int, default=4560, help="Sonic Pi OSC input port")
    p.add_argument("--gain", default="MAX", choices=["LOW", "MED", "HIGH", "MAX"])
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--wav", help="render data/run.csv to this WAV file instead of running live")
    p.add_argument("--seconds", type=float, default=60, help="length of the WAV render")
    args = p.parse_args()
    render_wav(args) if args.wav else live(args)


if __name__ == "__main__":
    main()
