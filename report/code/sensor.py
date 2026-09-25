"""Minimal TSL2591 driver over I2C (smbus), plus a simulator for dry runs.

Why not the Adafruit library? Its free-running mode returns the *last finished*
ADC value, so polling faster than the integration time silently repeats samples,
which ruins noise statistics. Here every read() restarts the ADC and waits for a
fresh integration (one-shot mode), so every sample is independent.

Register map (TSL2591 datasheet): command byte = 0xA0 | register.
  0x00 ENABLE  bit0 PON (oscillator on), bit1 AEN (ADC on)
  0x01 CONTROL bits5:4 gain, bits2:0 integration time (0=100 ms ... 5=600 ms)
  0x12 ID      reads 0x50
  0x13 STATUS  bit0 AVALID: an integration finished since AEN was set
  0x14..0x17   C0DATAL, C0DATAH (full spectrum), C1DATAL, C1DATAH (IR)
"""
import time

import numpy as np

ADDR = 0x29
CMD = 0xA0
REG_ENABLE, REG_CONTROL, REG_ID, REG_STATUS, REG_C0 = 0x00, 0x01, 0x12, 0x13, 0x14
PON, AEN = 0x01, 0x02
GAIN_BITS = {"LOW": 0x00, "MED": 0x10, "HIGH": 0x20, "MAX": 0x30}
GAIN_X = {"LOW": 1, "MED": 25, "HIGH": 428, "MAX": 9876}
ATIMES_MS = [100, 200, 300, 400, 500, 600]


def saturation(atime_ms):
    """Largest valid count; the 100 ms cycle saturates below 16 bits."""
    return 36863 if atime_ms == 100 else 65535


class TSL2591:
    simulated = False

    def __init__(self, gain="MAX", atime_ms=100, bus=1):
        try:
            from smbus2 import SMBus
        except ImportError:  # python3-smbus has the same calls
            from smbus import SMBus
        self.bus = SMBus(bus)
        dev_id = self.bus.read_byte_data(ADDR, CMD | REG_ID)
        if dev_id != 0x50:
            raise RuntimeError(f"chip at 0x29 reports ID 0x{dev_id:02X}, expected 0x50 (TSL2591)")
        self.gain, self.atime_ms = gain, atime_ms
        self.bus.write_byte_data(ADDR, CMD | REG_CONTROL,
                                 GAIN_BITS[gain] | ATIMES_MS.index(atime_ms))
        self.bus.write_byte_data(ADDR, CMD | REG_ENABLE, PON)
        time.sleep(0.005)

    def read(self):
        """Run one fresh integration. Returns (t_seconds, ch0_full, ch1_ir)."""
        self.bus.write_byte_data(ADDR, CMD | REG_ENABLE, PON)        # ADC off, clears AVALID
        self.bus.write_byte_data(ADDR, CMD | REG_ENABLE, PON | AEN)  # start a new cycle
        t0 = time.monotonic()
        time.sleep(1.12 * self.atime_ms / 1000)  # full cycle + margin, even if AVALID lags
        while not self.bus.read_byte_data(ADDR, CMD | REG_STATUS) & 0x01:
            if time.monotonic() - t0 > 2 * self.atime_ms / 1000 + 0.1:
                raise TimeoutError("TSL2591 never finished an integration")
            time.sleep(0.002)
        d = self.bus.read_i2c_block_data(ADDR, CMD | REG_C0, 4)  # one transaction latches all 4
        return time.monotonic(), d[0] | d[1] << 8, d[2] | d[3] << 8

    def close(self):
        self.bus.write_byte_data(ADDR, CMD | REG_ENABLE, 0x00)
        self.bus.close()


class SimulatedTSL2591:
    """Fake sensor with Poisson photon statistics, for rehearsing without hardware.
    Its numbers are invented: never put them in a report (analyze.py flags them)."""
    simulated = True

    def __init__(self, gain="MAX", atime_ms=100, realtime=False, seed=None):
        self.gain, self.atime_ms, self.realtime = gain, atime_ms, realtime
        self.rng = np.random.default_rng(seed)
        self.k = 3000.0 * 9876 / GAIN_X[gain]  # electrons per count (made up)
        self.level, self.drift, self.t = 0.0, 0.0, 0.0

    def set_level(self, counts):
        self.level, self.drift = float(counts), 0.0

    def read(self):
        dt = self.atime_ms / 1000
        if self.realtime:
            time.sleep(dt)
        self.t += dt * 1.05
        self.drift += self.rng.normal(0, 1e-4)  # slow source wander
        out = []
        for share, dark in ((1.0, 3.0), (0.35, 1.0)):
            mean_e = max(self.level * share * (1 + self.drift), 0) * self.k
            counts = self.rng.poisson(mean_e) / self.k + dark + self.rng.normal(0, 1.2)
            out.append(int(np.clip(round(counts), 0, saturation(self.atime_ms))))
        return self.t, out[0], out[1]

    def close(self):
        pass


def open_sensor(simulate=False, gain="MAX", atime_ms=100, realtime=False):
    if simulate:
        return SimulatedTSL2591(gain, atime_ms, realtime=realtime)
    return TSL2591(gain, atime_ms)
