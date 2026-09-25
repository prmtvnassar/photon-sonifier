# Photon-Noise Sonifier

TSL2591 light sensor + Raspberry Pi: one recording of phone light through a foil pinhole, turn the
sensor's noise into random numbers, play them as music, and generate the LaTeX report from the data.

```
python3 collect.py check      # wiring, aiming, light-leak test
python3 collect.py run        # the one recording (move the phone until it says "good", press Enter)
python3 analyze.py            # random bits, dice, tests; fills report/, writes report_overleaf.zip
python3 sonify.py             # live music (Sonic Pi running sonic_pi.rb)
python3 waves.py pluck        # math sound waves on the USB speaker (--device plughw:N,0)
```

Add `--simulate` to rehearse without hardware, then `rm -r data` before the real run.
