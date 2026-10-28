"""Arrangements of public-domain pieces, performed by small synthesised instruments.

The compositions are long out of copyright (Bach, Grieg) and the performance is
synthesised here, so the soundtrack carries no licence risk. Everything is written on a
beat grid at the show's tempo, so it lines up with the choreography bar for bar.

`score()` collects notes as (beat, beats, midi, instrument, velocity); `render()` plays
them, adds a hall reverb and returns stereo float32 at music.SR.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve, lfilter

from .music import SR, _crash, _note, _supersaw

# ---------------------------------------------------------------------------
# Instruments: f(freq, n_samples, velocity, rng) -> mono samples
# ---------------------------------------------------------------------------


def _env(n, a, r, hold=None):
    t = np.arange(n) / SR
    hold = n / SR - r if hold is None else hold
    e = np.minimum(1.0, t / max(a, 1e-4))
    return np.where(t > hold, e * np.exp(-(t - hold) / max(r / 4, 1e-4)), e)


def organ(f, n, v, rng):
    """Pipe organ: 16', 8', 4', 2 2/3', 2' ranks with a little chorus."""
    t = np.arange(n) / SR
    out = np.zeros(n)
    for mult, w in ((0.5, 0.55), (1, 1.0), (2, 0.6), (3, 0.3), (4, 0.25), (6, 0.12), (8, 0.08)):
        for det in (-0.0012, 0.0012):
            out += w * np.sin(2 * np.pi * f * mult * (1 + det) * t + rng.uniform(0, 6.28))
    return out * _env(n, 0.03, 0.35) * v * 0.12


def pizz(f, n, v, rng):
    """Plucked strings: harmonics that die away faster the higher they are."""
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k in range(1, 9):
        out += np.sin(2 * np.pi * f * k * t) / k * np.exp(-t * (5 + 2.5 * k))
    return out * np.minimum(1, t / 0.002) * v * 0.5


def reed(f, n, v, rng):
    """Bassoon-ish: a soft, nasal saw through a low-pass, with slight vibrato."""
    t = np.arange(n) / SR
    ph = np.cumsum(f * (1 + 0.004 * np.sin(2 * np.pi * 5.5 * t))) / SR
    x = 2 * (ph % 1.0) - 1
    x = lfilter([0.12], [1, -0.88], x)
    return np.tanh(2.5 * x) * _env(n, 0.015, 0.08) * v * 0.35


def strings(f, n, v, rng):
    """Section strings: detuned saws, filtered, bowed attack."""
    x = _supersaw(f, n, voices=5, spread=0.35, rng=rng)
    x = lfilter([0.18], [1, -0.82], x)
    return x * _env(n, 0.04, 0.25) * v * 0.55


def brass(f, n, v, rng):
    """Brass stab: a saw whose filter opens on the attack."""
    t = np.arange(n) / SR
    x = 2 * ((f * t + rng.uniform()) % 1.0) - 1
    a = 0.55 + 0.4 * np.exp(-t * 9)
    y = np.zeros(n)
    s = 0.0
    # One-pole low-pass with a time-varying coefficient, in blocks for speed.
    blk = 256
    for i in range(0, n, blk):
        c = a[i]
        seg = lfilter([1 - c], [1, -c], x[i:i + blk], zi=[s * c])
        y[i:i + blk], s = seg[0], seg[0][-1] if len(seg[0]) else s
    return np.tanh(2 * y) * _env(n, 0.01, 0.12) * v * 0.4


def timpani(f, n, v, rng):
    t = np.arange(n) / SR
    fr = f * (1 + 0.08 * np.exp(-t * 30))
    body = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 2.2)
    hit = lfilter([0.3], [1, -0.7], rng.standard_normal(n)) * np.exp(-t * 40)
    return (body + 0.6 * hit) * v * 0.9


def gong(f, n, v, rng):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for k, w in ((1, 1), (1.47, 0.6), (2.09, 0.5), (2.56, 0.35), (3.39, 0.25), (4.1, 0.2)):
        out += w * np.sin(2 * np.pi * f * k * t + rng.uniform(0, 6.28)) * np.exp(-t * (0.7 + 0.4 * k))
    swell = np.minimum(1, t / 0.05)
    return out * swell * v * 0.35


def crash(f, n, v, rng):
    return _crash(n, rng) * v * 2.0


def epiano(f, n, v, rng):
    """Electric piano: a bell-ish FM tone that decays."""
    t = np.arange(n) / SR
    mod = 1.4 * np.exp(-t * 7) * np.sin(2 * np.pi * f * t)
    return np.sin(2 * np.pi * f * t + mod) * np.exp(-t * 1.6) * v * 0.35


def clav(f, n, v, rng):
    t = np.arange(n) / SR
    x = 2 * ((f * t) % 1.0) - 1
    x = x - lfilter([0.3], [1, -0.7], x)            # thin, bright
    return x * np.exp(-t * 14) * v * 0.3


def bass(f, n, v, rng):
    t = np.arange(n) / SR
    x = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) + 0.15 * np.sin(6 * np.pi * f * t)
    return np.tanh(1.5 * x) * _env(n, 0.004, 0.05) * v * 0.55


def square(f, n, v, rng):
    t = np.arange(n) / SR
    return np.sign(np.sin(2 * np.pi * f * t)) * _env(n, 0.002, 0.03) * v * 0.16


def triangle(f, n, v, rng):
    t = np.arange(n) / SR
    return (2 * np.abs(2 * ((f * t) % 1.0) - 1) - 1) * _env(n, 0.002, 0.02) * v * 0.45


def blip(f, n, v, rng):
    t = np.arange(n) / SR
    return np.sin(2 * np.pi * f * t) * np.exp(-t * 18) * v * 0.4


def kick(f, n, v, rng):
    t = np.arange(n) / SR
    fr = 48 + 110 * np.exp(-t * 32)
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 7) * v


def snare(f, n, v, rng):
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t * 25)
    noise = lfilter([1, -1], [1, -0.6], rng.standard_normal(n)) * np.exp(-t * 18)
    return (0.5 * tone + 0.6 * noise) * v * 0.6


def hat(f, n, v, rng):
    t = np.arange(n) / SR
    noise = lfilter([1, -1], [1, 0.3], rng.standard_normal(n))
    dec = 60 if f < 1000 else 14                    # midi < ~84 closed, else open
    return noise * np.exp(-t * dec) * v * 0.18


def chipnoise(f, n, v, rng):
    t = np.arange(n) / SR
    x = np.repeat(rng.choice([-1.0, 1.0], n // 20 + 1), 20)[:n]
    return x * np.exp(-t * (40 if f < 1000 else 10)) * v * 0.2


def riser(f, n, v, rng):
    """Filtered noise that opens and swells over its length."""
    t = np.arange(n) / SR
    x = rng.standard_normal(n)
    y = np.zeros(n)
    blk = 512
    s = 0.0
    for i in range(0, n, blk):
        c = 0.98 - 0.9 * (i / n)
        seg, zf = lfilter([1 - c], [1, -c], x[i:i + blk], zi=[s])
        y[i:i + blk], s = seg, zf[0]
    return y * (t / (n / SR)) ** 2 * v * 0.8


def powerdown(f, n, v, rng):
    t = np.arange(n) / SR
    fr = f * np.exp(-t * 2.5)
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t * 1.2) * v * 0.4


INSTR = {"organ": organ, "pizz": pizz, "reed": reed, "strings": strings, "brass": brass,
         "timpani": timpani, "gong": gong, "crash": crash, "epiano": epiano, "clav": clav,
         "bass": bass, "square": square, "triangle": triangle, "blip": blip, "kick": kick,
         "snare": snare, "hat": hat, "chipnoise": chipnoise, "riser": riser,
         "powerdown": powerdown}
# Ring-out after a note ends, seconds.
TAIL = {"organ": 0.4, "pizz": 0.6, "reed": 0.1, "strings": 0.3, "brass": 0.15,
        "timpani": 1.8, "gong": 4.0, "crash": 2.5, "epiano": 1.5, "clav": 0.3, "bass": 0.08,
        "square": 0.05, "triangle": 0.05, "blip": 0.3, "kick": 0.5, "snare": 0.35, "hat": 0.35,
        "chipnoise": 0.3, "riser": 0.05, "powerdown": 2.0}
PAN = {"organ": 0.0, "pizz": -0.25, "reed": 0.2, "strings": -0.15, "brass": 0.25,
       "timpani": 0.1, "gong": 0.0, "crash": 0.3, "epiano": -0.2, "clav": 0.3, "bass": 0.0,
       "square": 0.1, "triangle": 0.0, "blip": 0.2, "kick": 0.0, "snare": 0.05, "hat": 0.25,
       "chipnoise": -0.1, "riser": 0.0, "powerdown": 0.0}


def render(notes, bpm, seconds, reverb_s=2.4, wet=0.28, seed=11):
    """notes: (beat, beats, midi, instrument, velocity). Returns (n, 2) float32."""
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    N = int(seconds * SR)
    L = np.zeros(N)
    R = np.zeros(N)
    for b0, nb, m, ins, v in notes:
        i0 = int(b0 * beat * SR)
        if i0 >= N:
            continue
        n = int((nb * beat + TAIL[ins]) * SR)
        x = INSTR[ins](_note(m), n, v, rng)
        if ins in ("organ", "strings", "reed", "brass", "bass", "square", "triangle"):
            # Hold for the written length, then release.
            x *= _env(n, 0.001, TAIL[ins] * 4, hold=nb * beat)
        x = x[:N - i0]
        p = PAN[ins]
        L[i0:i0 + len(x)] += x * (1 - p)
        R[i0:i0 + len(x)] += x * (1 + p)
    # Hall: two decorrelated decaying-noise impulse responses.
    ir_n = int(reverb_s * SR)
    tt = np.arange(ir_n) / SR
    irs = []
    for k in range(2):
        ir = np.random.default_rng(seed + k).standard_normal(ir_n) * np.exp(-tt * 6.9 / reverb_s)
        ir = lfilter([0.5], [1, -0.5], ir)
        irs.append(ir / np.sqrt((ir ** 2).sum()))
    L = L + wet * fftconvolve(L, irs[0])[:N]
    R = R + wet * fftconvolve(R, irs[1])[:N]
    out = np.stack([L, R], 1)
    out /= max(1e-9, np.percentile(np.abs(out), 99.9)) / 0.8
    return np.tanh(out * 1.1).astype(np.float32) * 0.9


# ---------------------------------------------------------------------------
# Pieces
# ---------------------------------------------------------------------------

# Grieg, "In the Hall of the Mountain King" (1875): the two phrases, as
# (midi, beats) with eighth notes = half a beat. Phrase A in B minor.
MK_A = [(59, .5), (61, .5), (62, .5), (64, .5), (66, .5), (62, .5), (66, 1),
        (65, .5), (61, .5), (65, 1), (64, .5), (60, .5), (64, 1),
        (59, .5), (61, .5), (62, .5), (64, .5), (66, .5), (62, .5), (66, .5), (71, .5),
        (69, .5), (66, .5), (62, .5), (66, .5), (69, 2)]
# Phrase B, on the dominant.
MK_B = [(66, .5), (68, .5), (70, .5), (71, .5), (73, .5), (70, .5), (73, 1),
        (74, .5), (70, .5), (74, 1), (73, .5), (70, .5), (73, 1),
        (66, .5), (68, .5), (70, .5), (71, .5), (73, .5), (70, .5), (73, .5), (78, .5),
        (77, .5), (73, .5), (70, .5), (73, .5), (77, 2)]


def _line(notes, phrase, start, ins, vel, shift=0, octaves=(0,), stacc=1.0, upto=None):
    b = start
    for m, d in phrase:
        if upto is not None and b >= upto:
            break
        for o in octaves:
            notes.append((b, d * stacc, m + shift + 12 * o, ins, vel))
        b += d
    return b


def halloween(bpm=128.0, seconds=30.0):
    """Toccata in D minor (Bach) opening, transposed to B minor, into the Mountain King.
    Laid out on the Halloween show: intro 2 bars, sneak 4, ignite 2, flee 6, end 2."""
    n = []
    # Bars 0-2: the Toccata mordent and run on full organ, a pedal B under it.
    for b, d, m in ((0, .25, 78), (.25, .25, 76), (.5, 1.75, 78),
                    (2.5, .25, 76), (2.75, .25, 74), (3.0, .25, 73), (3.25, .25, 71),
                    (3.5, .75, 70), (4.25, 1.75, 71)):
        for o in (0, -12):
            n.append((b, d, m + o, "organ", 0.9))
    for m in (35, 47, 59, 62, 66):                  # B minor with a low pedal
        n.append((4.25, 1.75, m, "organ", 0.7))
    n.append((4.25, 0, 35, "gong", 0.6))
    for b, d, m in ((6.0, .25, 66), (6.25, .25, 64), (6.5, .5, 66)):
        n.append((b, d, m - 12, "organ", 0.8))
    for k, m in enumerate((58, 61, 64, 67, 70)):     # A# diminished seventh climbing
        n.append((7.0 + 0.2 * k, 1.0 - 0.2 * k, m, "organ", 0.6 + 0.08 * k))
    # Bars 2-6: phrase A sneaking in on pizzicato and bassoon.
    _line(n, MK_A, 8, "pizz", 0.55, shift=-12)
    _line(n, MK_A, 8, "reed", 0.35, shift=-12, stacc=0.5)
    for b in range(8, 24, 4):
        n.append((b, 0, 47, "timpani", 0.15))
    # Bars 6-8: the face ignites. Phrase B, louder, strings in, timpani roll up.
    n.append((24, 0, 0, "crash", 0.5))
    n.append((24, 0, 35, "gong", 0.5))
    _line(n, MK_B, 24, "pizz", 0.7, shift=-12, upto=32)
    _line(n, MK_B, 24, "strings", 0.45, shift=0, stacc=0.6, upto=32)
    _line(n, MK_B, 24, "reed", 0.45, shift=-12, stacc=0.5, upto=32)
    for k in range(32):
        n.append((28 + k * 0.125, 0, 42, "timpani", 0.15 + 0.6 * k / 32))
    # Bars 8-12: everyone runs. Phrase A tutti.
    n.append((32, 0, 0, "crash", 0.9))
    _line(n, MK_A, 32, "strings", 0.8, octaves=(0, 1), stacc=0.7)
    _line(n, MK_A, 32, "brass", 0.6, shift=-12, stacc=0.45)
    _line(n, MK_A, 32, "organ", 0.35, shift=-12, stacc=0.8)
    for b in range(32, 48):
        n.append((b, 0, 47 if b % 2 == 0 else 42, "timpani", 0.4))
    # Bars 12-14: BOO! Phrase B tutti, up an octave.
    n.append((48, 0, 0, "crash", 0.9))
    _line(n, MK_B, 48, "strings", 0.85, octaves=(0, 1), stacc=0.7, upto=56)
    _line(n, MK_B, 48, "brass", 0.7, shift=-12, stacc=0.45, upto=56)
    for b in range(48, 56):
        n.append((b, 0, 42 if b % 2 else 47, "timpani", 0.45))
    # Bars 14-15: B minor hammer blows on every beat, then the final chord.
    for k, b in enumerate(range(56, 60)):
        for m in (47, 59, 62, 66, 71):
            n.append((b, 0.4, m, "brass", 0.75))
            n.append((b, 0.4, m, "strings", 0.6))
        n.append((b, 0, 35, "timpani", 0.9))
    for m in (35, 47, 59, 62, 66, 71, 74):
        n.append((60, 4, m, "organ", 0.9))
    n.append((60, 0, 0, "crash", 1.0))
    n.append((60, 0, 35, "gong", 1.0))
    return render(n, bpm, seconds, reverb_s=2.8, wet=0.32)


# ---------------------------------------------------------------------------
# Originals for the one-crab hero: intro 4 bars (the robot boots, eyes open, smiles),
# build 2 (the hero dances alone), drop 8 (everyone), outro 2 (wink, power down).
# ---------------------------------------------------------------------------

HOOK = [(0, .5, 69), (.5, .5, 72), (1, .25, 74), (1.25, .75, 76), (2, .5, 79), (2.5, .5, 76),
        (3, .5, 74), (3.5, .5, 72),
        (4, .75, 74), (4.75, .25, 72), (5, 1, 69), (6.5, .5, 67), (7, 1, 69)]
CHORDS = [(57, 60, 64, 67), (62, 66, 69, 72)]       # Am7 | D9-ish: A dorian
BASSROOT = [45, 50]


def _boot(n, at):
    """The robot wakes: a rising chime, then a happy upward bloop."""
    for k, m in enumerate((72, 76, 79, 84)):
        n.append((at + 0.12 * k, 0.3, m, "blip", 0.6))


def disco(bpm=128.0, seconds=30.0):
    n = []
    _boot(n, 1 * 4 * 0.75 / 1.0 * 1.0)              # bar 0.75
    n.append((6.0, .4, 81, "blip", .5)); n.append((6.15, .5, 88, "blip", .5))   # blink
    n.append((10.0, .3, 84, "blip", .6)); n.append((10.2, .6, 91, "blip", .6))  # happy
    for bar in range(16):
        b = bar * 4
        ch = CHORDS[bar % 2]
        root = BASSROOT[bar % 2]
        if bar < 4:                                   # intro: soft electric piano
            for m in ch:
                n.append((b, 3.5, m, "epiano", 0.35 + 0.08 * bar))
        if 4 <= bar < 6:                              # build: bass, hats, claps, riser
            for k in range(8):
                n.append((b + k * .5, .4, root + (12 if k % 2 else 0), "bass", 0.6))
            for k in range(8):
                n.append((b + k * .5 + .5 * (k % 2 == 0), 0, 90, "hat", 0.5))
            n.append((b + 1, 0, 0, "snare", 0.5)); n.append((b + 3, 0, 0, "snare", 0.5))
            for m in ch:
                n.append((b, 3.5, m, "epiano", 0.5))
        if bar == 4:
            n.append((16, 8, 0, "riser", 0.7))
        if bar == 5:
            for k in range(8):                        # snare fill into the drop
                n.append((22 + k * .25, 0, 0, "snare", 0.4 + 0.07 * k))
        if 6 <= bar < 14:                             # drop: the full groove
            for k in range(4):
                n.append((b + k, 0, 0, "kick", 1.0))
                n.append((b + k + .5, 0, 90, "hat", 0.8))
                n.append((b + k + .25, 0, 70, "hat", 0.35))
                n.append((b + k + .75, 0, 70, "hat", 0.35))
            n.append((b + 1, 0, 0, "snare", 0.8)); n.append((b + 3, 0, 0, "snare", 0.8))
            for k in range(8):
                n.append((b + k * .5, .4, root + (12 if k % 2 else 0), "bass", 0.8))
            for off in (.5, 1.25, 1.75, 2.5, 3.25, 3.75):
                for m in ch[1:]:
                    n.append((b + off, .15, m + 12, "clav", 0.55))
            for m in ch:
                n.append((b + 1.5, .5, m + 12, "strings", 0.4))
            if bar in (6, 9, 12):                     # brass hits on the big jumps
                n.append((b, 0, 0, "crash", 0.7))
                for m in ch:
                    n.append((b, .5, m, "brass", 0.7))
        if 8 <= bar < 14 and bar % 2 == 0:            # the hook, twice
            for o, d, m in HOOK:
                n.append((b + o, d, m + 12, "brass", 0.55))
                n.append((b + o, d, m, "square", 0.35))
        if bar >= 14:                                 # outro: piano, a wink, power down
            for m in ch:
                n.append((b, 3.5, m, "epiano", 0.45))
    n.append((56, 0, 0, "kick", 0.9)); n.append((56, 0, 0, "crash", 0.6))
    n.append((59.0, .3, 88, "blip", .6))                                     # wink
    n.append((62.4, 1.5, 76, "powerdown", 0.9))                              # eyes off
    return render(n, bpm, seconds, reverb_s=1.4, wet=0.15)


def chiptune(bpm=128.0, seconds=30.0):
    """8-bit, laid out on the one-crab hero: boot 0-3, zoom in 3-5 (build), drop at 5 as
    the camera reaches the hero, dance spreads 5-8, the picture shatters 8-10.5, tiles
    mix 8-14 (a hop per bar), reform at 14, smile, wink, cut at 15.85."""
    n = []
    _boot(n, 3.0)                                             # eyes boot (bar 0.75)
    n.append((6.0, .2, 81, "blip", .5)); n.append((6.2, .3, 88, "blip", .5))     # blink
    n.append((10.0, .3, 84, "blip", .6)); n.append((10.2, .6, 91, "blip", .6))   # smile
    n.append((12.0, .3, 79, "blip", .5)); n.append((12.15, .4, 86, "blip", .5))  # eyes open
    for bar in range(16):
        b = bar * 4
        ch = CHORDS[bar % 2]
        root = BASSROOT[bar % 2]
        groove = 5 <= bar < 14
        if bar < 3:                                           # intro: gentle arpeggio
            for k in range(8):
                n.append((b + k * .5, .45, ch[k % 4] + 12, "square", 0.3 + 0.05 * bar))
        if 3 <= bar < 5:                                      # zoom in: build
            step = .25
            for k in range(16):
                rise = (bar - 3) * 4 + k * .25
                n.append((b + k * step, .22, ch[k % 4] + 12 + (12 if rise >= 4 else 0),
                          "square", 0.35 + 0.04 * rise))
            for k in range(8):
                n.append((b + k * .5, .45, root + (12 if k % 2 else 0), "triangle", 0.6))
        if bar == 4:
            for k in range(16):                               # noise roll into the drop
                n.append((16 + k * .25, 0, 1200, "chipnoise", 0.3 + 0.04 * k))
            for k in range(16):                               # rising sweep
                n.append((16 + k * .25, .25, 60 + k * 2, "square", 0.12 + 0.01 * k))
        if bar == 5:
            n.append((20, 0, 1200, "chipnoise", 1.0)); n.append((20, 0, 1200, "crash", 0.6))
        if groove:
            for k in range(8):
                n.append((b + k * .5, .45, root + (12 if k % 2 else 0), "triangle", 0.8))
            for k in range(4):
                n.append((b + k, .1, 36, "triangle", 1.0))            # chip kick
                n.append((b + k + .5, 0, 70, "chipnoise", 0.6))       # hat
            n.append((b + 1, 0, 1200, "chipnoise", 0.8)); n.append((b + 3, 0, 1200, "chipnoise", 0.8))
            for off in (0, .75, 1.5, 2.25, 3):
                for m in ch[1:3]:
                    n.append((b + off, .2, m, "square", 0.22))
        if bar in (5, 7, 10, 12):                            # the hook
            for o, d, m in HOOK:
                n.append((b + o, d * .9, m + 12, "square", 0.55))
                n.append((b + o + .02, d * .9, m + 24, "square", 0.15))
        if 8 <= bar < 14:                                     # the mix: a hop on every step
            for k, m in enumerate((84, 88, 91)):
                n.append((b + k * .12, .15, m - (bar % 2) * 2, "blip", 0.45))
        if bar in (8, 9):                                     # answer phrase while they wander
            for o, d, m in ((0, .5, 76), (.5, .5, 74), (1, 1, 72), (2, .5, 69), (2.5, .5, 72), (3, 1, 74)):
                n.append((b + o, d * .9, m + 12, "square", 0.4))
        if 11 <= bar < 14:                                    # coming back together: build
            for k in range(8):
                n.append((b + k * .5, .4, ch[k % 4] + 24, "square", 0.15 + 0.03 * (bar - 11) * 2))
        if bar == 13:
            for k in range(16):
                n.append((52 + k * .25, 0, 1200, "chipnoise", 0.3 + 0.04 * k))
        if bar >= 14:                                         # outro, cut dead at the end
            for k in range(8):
                if b + k * .5 < 63.4:
                    n.append((b + k * .5, .45, ch[k % 4] + 12, "square", 0.32))
    n.append((56, 0, 1200, "crash", 0.8)); n.append((56, .1, 36, "triangle", 1.0))  # reformed
    for k in range(10):                                        # the magic sparkle
        n.append((56 + k * .06, .12, 84 + 2 * k, "blip", 0.45))
    n.append((58.0, .3, 84, "blip", .6)); n.append((58.2, .6, 91, "blip", .6))   # smile
    n.append((60.0, .3, 88, "blip", .6)); n.append((60.25, .2, 93, "blip", .5))  # wink
    n.append((63.4, .1, 36, "triangle", 1.0)); n.append((63.4, 0, 1200, "chipnoise", 0.9))  # off
    n = [x for x in n if x[0] < 63.4 or x[3] in ("triangle", "chipnoise")]
    return render(n, bpm, seconds, reverb_s=0.8, wet=0.08)


def chiptune_v4(bpm=128.0, seconds=33.75):
    """8-bit, on the v4 hero: boot and smile 0-3, zoom 3-5 (build), drop at 5 into the
    party 5-10, marching-band mix 10-15 (a hop per bar), reform at 15, settle, wink at
    17, cut at 17.9."""
    n = []
    _boot(n, 3.0)
    n.append((6.0, .2, 81, "blip", .5)); n.append((6.2, .3, 88, "blip", .5))     # blink
    n.append((8.0, .3, 79, "blip", .5)); n.append((8.15, .4, 86, "blip", .5))    # eyes open
    n.append((10.0, .3, 84, "blip", .6)); n.append((10.2, .6, 91, "blip", .6))   # smile
    end = 17.9 * 4
    for bar in range(18):
        b = bar * 4
        ch = CHORDS[bar % 2]
        root = BASSROOT[bar % 2]
        if bar < 3 or 15 <= bar < 17:                         # calm arpeggio
            for k in range(8):
                n.append((b + k * .5, .45, ch[k % 4] + 12, "square", 0.3 + 0.03 * (bar % 3)))
        if bar == 17:
            for k in range(8):
                if b + k * .5 < end:
                    n.append((b + k * .5, .45, ch[k % 4] + 12, "square", 0.3))
        if 3 <= bar < 5:                                      # zoom in: build
            for k in range(16):
                rise = (bar - 3) * 4 + k * .25
                n.append((b + k * .25, .22, ch[k % 4] + 12 + (12 if rise >= 4 else 0),
                          "square", 0.35 + 0.04 * rise))
            for k in range(8):
                n.append((b + k * .5, .45, root + (12 if k % 2 else 0), "triangle", 0.6))
        if bar == 4:
            for k in range(16):
                n.append((16 + k * .25, 0, 1200, "chipnoise", 0.3 + 0.04 * k))
                n.append((16 + k * .25, .25, 60 + k * 2, "square", 0.12 + 0.01 * k))
        if bar == 5:
            n.append((20, 0, 1200, "chipnoise", 1.0)); n.append((20, 0, 1200, "crash", 0.6))
        if 5 <= bar < 15:                                     # the groove
            for k in range(8):
                n.append((b + k * .5, .45, root + (12 if k % 2 else 0), "triangle", 0.8))
            for k in range(4):
                n.append((b + k, .1, 36, "triangle", 1.0))
                n.append((b + k + .5, 0, 70, "chipnoise", 0.6))
            n.append((b + 1, 0, 1200, "chipnoise", 0.8)); n.append((b + 3, 0, 1200, "chipnoise", 0.8))
            for off in (0, .75, 1.5, 2.25, 3):
                for m in ch[1:3]:
                    n.append((b + off, .2, m, "square", 0.22))
        if bar in (5, 7, 12):                                 # the hook
            for o, d, m in HOOK:
                n.append((b + o, d * .9, m + 12, "square", 0.55))
                n.append((b + o + .02, d * .9, m + 24, "square", 0.15))
        if 5 <= bar < 10:                                     # party: jump hits each bar
            n.append((b, .15, 79, "blip", 0.35)); n.append((b + 2, .15, 83, "blip", 0.3))
        if 10 <= bar < 15:                                    # marching band: rudiments + hop
            for k in (0, .25, .5, 2, 2.25, 2.5, 3.5, 3.75):
                n.append((b + k, 0, 1200, "chipnoise", 0.35))
            for k, m in enumerate((84, 88, 91)):
                n.append((b + k * .12, .15, m - (bar % 2) * 2, "blip", 0.45))
        if bar in (10, 11):
            for o, d, m in ((0, .5, 76), (.5, .5, 74), (1, 1, 72), (2, .5, 69), (2.5, .5, 72), (3, 1, 74)):
                n.append((b + o, d * .9, m + 12, "square", 0.4))
        if bar == 14:
            for k in range(16):
                n.append((56 + k * .25, 0, 1200, "chipnoise", 0.3 + 0.04 * k))
    n.append((60, 0, 1200, "crash", 0.8)); n.append((60, .1, 36, "triangle", 1.0))   # reformed
    for k in range(10):
        n.append((60 + k * .06, .12, 84 + 2 * k, "blip", 0.45))
    n.append((66.0, .3, 79, "blip", .5))                                          # eyes open
    n.append((68.0, .2, 88, "blip", .6)); n.append((68.25, .3, 95, "blip", .6))   # wink
    n.append((69.6, .3, 84, "blip", .6)); n.append((69.8, .5, 91, "blip", .6))    # smile
    n.append((end, .1, 36, "triangle", 1.0)); n.append((end, 0, 1200, "chipnoise", 0.9))
    n = [x for x in n if x[0] <= end]
    return render(n, bpm, seconds, reverb_s=0.8, wet=0.08)


PIECES = {"halloween": halloween, "disco": disco, "chiptune": chiptune, "chiptune_v4": chiptune_v4}
