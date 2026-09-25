"""Data collection for the photon-noise sonifier.

  python3 collect.py check          live readout (wiring, aiming, light-leak test)
  python3 collect.py run            the one recording: aim the phone, press Enter, wait

Add --simulate to rehearse without hardware.
"""
import argparse
import csv
import os
import select
import sys

from sensor import open_sensor, saturation

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
TARGET = (3000, 25000)  # good ch0 range: plenty of noise, far from saturation


def enter_pressed():
    return bool(select.select([sys.stdin], [], [], 0)[0])


def cmd_check(sensor, args):
    print(f"TSL2591 OK  gain={sensor.gain}  integration={sensor.atime_ms} ms  "
          f"saturation={saturation(sensor.atime_ms)}   (Ctrl+C to stop)")
    try:
        while True:
            _, c0, c1 = sensor.read()
            bar = "#" * min(60, int(60 * c0 / saturation(sensor.atime_ms)))
            print(f"  ch0 {c0:6d}  ch1 {c1:6d}  |{bar}")
    except KeyboardInterrupt:
        print()


def aim(sensor):
    """Live readout until Enter, with a hint to move the phone closer or farther."""
    print(f"\nPoint the phone flashlight at the hole. Move it closer/farther until ch0 is "
          f"between {TARGET[0]} and {TARGET[1]}, rest the phone, then press Enter.")
    sat = saturation(sensor.atime_ms)
    while True:
        _, c0, c1 = sensor.read()
        hint = ("SATURATED - move farther" if c0 >= 0.95 * sat else "move closer" if c0 < TARGET[0]
                else "move farther" if c0 > TARGET[1] else "good - press Enter")
        sys.stdout.write(f"\r  ch0 {c0:6d}  ch1 {c1:6d}   {hint:<26}")
        sys.stdout.flush()
        if enter_pressed():
            sys.stdin.readline()
            print()
            return


def cmd_run(sensor, args):
    if sensor.simulated:
        sensor.set_level(args.sim_level)
    else:
        aim(sensor)
    n = int(args.minutes * 60 / (sensor.atime_ms / 1000 * 1.15))
    print(f"Recording {args.minutes:g} min ({n} samples). Don't touch anything.")
    rows = []
    for i in range(n):
        t, c0, c1 = sensor.read()
        rows.append((t, c0, c1))
        if i % 50 == 0:
            sys.stdout.write(f"\r  {i}/{n}  ch0 {c0:6d}")
            sys.stdout.flush()
    print(f"\r  {n}/{n} done            ")
    os.makedirs(DATA, exist_ok=True)
    path = os.path.join(DATA, "run.csv")
    t0 = rows[0][0]
    with open(path, "w", newline="") as f:
        f.write(f"# gain={sensor.gain} atime_ms={sensor.atime_ms} simulated={int(sensor.simulated)}\n")
        w = csv.writer(f)
        w.writerow(["t_s", "ch0", "ch1"])
        w.writerows((f"{t - t0:.4f}", c0, c1) for t, c0, c1 in rows)
    print(f"Saved data/run.csv.  Next: python3 analyze.py")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check", "run"])
    p.add_argument("--minutes", type=float, default=5, help="length of the recording")
    p.add_argument("--gain", default="MAX", choices=["LOW", "MED", "HIGH", "MAX"])
    p.add_argument("--simulate", action="store_true", help="fake sensor, for rehearsal only")
    p.add_argument("--sim-level", type=float, default=12000, help=argparse.SUPPRESS)
    args = p.parse_args()
    sensor = open_sensor(args.simulate, args.gain, 100, realtime=args.command == "check")
    try:
        {"check": cmd_check, "run": cmd_run}[args.command](sensor, args)
    finally:
        sensor.close()


if __name__ == "__main__":
    main()
