# Photon-noise sonifier: paste into a Sonic Pi buffer, press Run, then start sonify.py.
# Notes arrive as OSC /photon/note [midi, duration, amp]; the light level as /photon/level [0..1].

set :cut, 90

live_loop :notes do
  use_real_time
  n, dur, amp = sync "/osc*/photon/note"
  with_fx :reverb, room: 0.8, mix: 0.4 do
    synth :pretty_bell, note: n, release: dur, amp: amp
  end
end

live_loop :level do
  use_real_time
  lvl = sync "/osc*/photon/level"
  set :cut, 55 + 75 * lvl[0]   # brighter light -> brighter pad
end

live_loop :pad do
  synth :hollow, note: :a2, attack: 2, sustain: 2, release: 4, cutoff: get(:cut), amp: 0.7
  sleep 4
end
