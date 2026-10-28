"""Music: analyse any song for beats and drops, and synthesise an original track.

The posted videos use `synth_track` (written here, so no licence questions). `analyse`
works on any audio file, which is what "dance to any song" runs on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SR = 44100


@dataclass
class Analysis:
    tempo: float
    beats: np.ndarray        # seconds
    downbeats: np.ndarray    # seconds, every 4th beat aligned to the strongest phase
    energy_t: np.ndarray     # seconds
    energy: np.ndarray       # 0..1 smoothed loudness
    drops: np.ndarray        # seconds where energy jumps sharply
    duration: float


def analyse(path: str) -> Analysis:
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    # Downbeat phase: the one of four offsets whose beats carry the most onset strength.
    strength = onset[np.clip(beat_frames, 0, len(onset) - 1)]
    phase = int(np.argmax([strength[k::4].sum() for k in range(4)]))
    rms = librosa.feature.rms(y=y)[0]
    t = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    k = max(1, int(0.5 * sr / 512))
    smooth = np.convolve(rms, np.ones(k) / k, mode="same")
    smooth = (smooth - smooth.min()) / (np.ptp(smooth) + 1e-9)
    lag = int(2.0 * sr / 512)
    jump = np.zeros_like(smooth)
    jump[lag:] = smooth[lag:] - smooth[:-lag]
    drops = t[(jump > 0.35) & (np.r_[0, np.diff(np.sign(np.diff(jump))), 0] < 0)]
    return Analysis(float(np.atleast_1d(tempo)[0]), beats, beats[phase::4], t, smooth,
                    drops, len(y) / sr)


# ---------------------------------------------------------------------------
# Original track synthesis: a 128-BPM build-and-drop, deterministic from a seed.
# ---------------------------------------------------------------------------

def _env(n, attack, decay):
    t = np.arange(n) / SR
    return np.minimum(t / max(attack, 1e-4), 1.0) * np.exp(-t / decay)


def _kick(n):
    t = np.arange(n) / SR
    f = 50 + 120 * np.exp(-t * 30)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.001, 0.25)


def _hat(n, rng):
    return rng.standard_normal(n) * _env(n, 0.0005, 0.03) * 0.25


def _clap(n, rng):
    noise = rng.standard_normal(n)
    return noise * _env(n, 0.002, 0.09) * 0.4


def _saw(freq, n):
    t = np.arange(n) / SR
    out = np.zeros(n)
    for detune in (-0.12, 0.0, 0.12):
        ph = (freq * (1 + detune / 100) * t) % 1.0
        out += 2 * ph - 1
    return out / 3


def _lowpass(x, cutoff):
    from scipy.signal import lfilter

    a = np.exp(-2 * np.pi * cutoff / SR)
    return lfilter([1 - a], [1, -a], x)


def synth_track(sections, bpm=128.0, seed=7):
    """sections: list of (name, bars) with name in {"intro","build","drop","break","outro"}.

    Returns (audio float32 stereo, section start times in seconds).
    """
    rng = np.random.default_rng(seed)
    beat = 60.0 / bpm
    bar_n = int(round(4 * beat * SR))
    beat_n = bar_n // 4
    total = sum(b for _, b in sections) * bar_n
    mix = np.zeros(total + SR)
    # A minor: Am - F - C - G
    chords = [[220.0, 261.63, 329.63], [174.61, 220.0, 261.63],
              [196.0, 261.63, 329.63], [196.0, 246.94, 293.66]]
    bass_roots = [55.0, 43.65, 65.41, 49.0]
    kick, hat, clap = _kick(beat_n), _hat(beat_n // 2, rng), _clap(beat_n, rng)
    starts = []
    pos = 0
    bar_index = 0
    for name, bars in sections:
        starts.append((name, pos / SR))
        for b in range(bars):
            c = chords[bar_index % 4]
            root = bass_roots[bar_index % 4]
            prog = (b + 1) / bars
            pad = sum(_saw(f, bar_n) for f in c) / 3
            cutoff = {"intro": 600, "build": 500 + 3500 * prog, "drop": 5000,
                      "break": 400, "outro": 800 * (1 - prog) + 200}[name]
            pad = _lowpass(pad, cutoff) * (0.18 if name != "drop" else 0.22)
            seg = mix[pos:pos + bar_n]
            seg += pad
            for k in range(4):
                o = k * beat_n
                if name in ("drop", "outro") or (name == "intro" and b >= bars // 2):
                    seg[o:o + beat_n] += kick * 0.9
                if name == "drop":
                    seg[o:o + beat_n] += 0.5 * _saw(root, beat_n) * _env(beat_n, 0.01, 0.12) \
                        * (np.arange(beat_n) > beat_n // 2)
                    seg[o + beat_n // 2:o + beat_n // 2 + len(hat)] += hat
                    if k % 2 == 1:
                        seg[o:o + beat_n] += clap
                if name == "build":
                    sub = 1 if prog <= 0.5 else (2 if prog <= 0.75 else 4)
                    for s in range(sub):
                        so = o + s * beat_n // sub
                        seg[so:so + len(clap) // sub] += clap[:len(clap) // sub] * (0.4 + prog)
                if name == "drop" and k == 0:
                    lead_f = c[2] * 2
                    arp = np.concatenate([_saw(f * 2, beat_n // 4) * _env(beat_n // 4, 0.002, 0.08)
                                          for f in (c + [lead_f])])
                    seg[o:o + len(arp)] += arp * 0.12
            if name == "build" and b == bars - 1:
                # Riser into the drop.
                t = np.arange(bar_n) / SR
                f = 200 + 1800 * (t / t[-1]) ** 2
                seg += 0.08 * np.sin(2 * np.pi * np.cumsum(f) / SR) * (t / t[-1])
            pos += bar_n
            bar_index += 1
    mix = mix[:total]
    mix /= np.abs(mix).max() + 1e-9
    mix = np.tanh(mix * 1.4) * 0.85
    stereo = np.stack([mix, mix], 1).astype(np.float32)
    return stereo, starts


def write_wav(path, audio):
    import wave

    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(audio.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------
# Rave track v2: sidechained supersaws, a plucked hook with delay, snare-roll
# build with a riser, crash on the drop. Original composition, deterministic.
# ---------------------------------------------------------------------------

def _note(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def _supersaw(freq, n, voices=7, spread=0.25, rng=None):
    t = np.arange(n) / SR
    out = np.zeros(n)
    rng = rng or np.random.default_rng(0)
    for v in range(voices):
        d = (v - (voices - 1) / 2) / ((voices - 1) / 2) * spread / 100 * 2
        ph = rng.uniform(0, 1)
        out += 2 * ((freq * (1 + d) * t + ph) % 1.0) - 1
    return out / voices


def _adsr(n, a, d, s, r, sus_frac=0.7):
    t = np.arange(n) / SR
    T = n / SR
    env = np.where(t < a, t / max(a, 1e-4),
                   np.where(t < a + d, 1 - (1 - s) * (t - a) / max(d, 1e-4), s))
    rel_start = T * sus_frac
    env = np.where(t > rel_start, env * np.exp(-(t - rel_start) / max(r, 1e-4)), env)
    return env


def _crash(n, rng):
    noise = rng.standard_normal(n)
    from scipy.signal import lfilter
    hp = lfilter([1, -1], [1, -0.97], noise)
    return hp * np.exp(-np.arange(n) / SR / 1.2) * 0.35


def synth_rave(sections, bpm=125.0, seed=3):
    """sections: list of (name, bars), names in intro/build/drop/break/outro.
    Returns (stereo float32, section starts)."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(seed)
    beat_n = int(round(60.0 / bpm * SR))
    bar_n = beat_n * 4
    six = beat_n // 4
    total = sum(b for _, b in sections) * bar_n
    L = np.zeros(total + 4 * SR)
    R = np.zeros_like(L)
    side = np.ones_like(L)          # sidechain gain, ducked by every kick

    # Am - F - C - G, voiced high; the hook outlines them.
    prog = [[57, 60, 64, 69], [53, 57, 60, 65], [55, 60, 64, 67], [55, 59, 62, 67]]
    bass = [33, 29, 36, 31]
    # A 2-bar hook in 16ths (MIDI, None = rest), repeated with the chord root shifting.
    hook = [81, None, 76, 79, None, 81, 84, None, 83, None, 79, 76, None, 79, 81, None,
            81, None, 76, 79, None, 81, 88, None, 86, 84, None, 83, 79, None, 76, None]

    kick = _kick(beat_n)
    clap = _clap(beat_n, rng)
    hat = _hat(six, rng)
    starts = []
    pos = 0
    gbar = 0
    for name, bars in sections:
        starts.append((name, pos / SR))
        for b in range(bars):
            prog_i = gbar % 4
            prog_frac = (b + 1) / bars
            o = pos
            seg = slice(o, o + bar_n)
            full = name in ("drop",)
            # Kick + sidechain
            kick_on = name in ("drop", "outro") or (name == "intro" and b >= bars // 2) \
                or (name == "build" and prog_frac <= 0.75)
            for k in range(4):
                ko = o + k * beat_n
                if kick_on:
                    L[ko:ko + beat_n] += kick * 0.95
                    R[ko:ko + beat_n] += kick * 0.95
                    duck = 1 - 0.75 * np.exp(-np.arange(beat_n) / SR / 0.09)
                    side[ko:ko + beat_n] = np.minimum(side[ko:ko + beat_n], duck)
            # Hats on the off-beats, claps on 2 and 4
            if name in ("drop", "build", "outro") or (name == "intro" and b >= 1):
                for k in range(4):
                    ho = o + k * beat_n + beat_n // 2
                    L[ho:ho + six] += hat * 1.1
                    R[ho:ho + six] += hat * 0.9
            if full or name == "outro":
                for k in (1, 3):
                    co = o + k * beat_n
                    L[co:co + beat_n] += clap * 0.8
                    R[co:co + beat_n] += clap * 0.8
            # Build: snare roll accelerating 4 -> 8 -> 16 -> 32 per bar, riser
            if name == "build":
                div = 4 if prog_frac <= 0.25 else 8 if prog_frac <= 0.5 else 16 if prog_frac <= 0.75 else 32
                step = bar_n // div
                for k in range(div):
                    so = o + k * step
                    m = min(len(clap), step)
                    L[so:so + m] += clap[:m] * (0.25 + 0.6 * prog_frac)
                    R[so:so + m] += clap[:m] * (0.25 + 0.6 * prog_frac)
                t = np.arange(bar_n) / SR
                f = 300 * 2 ** ((b + t / t[-1]) / bars * 3)
                riser = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.05 * (b + 1) / bars
                noise = lfilter([1, -1], [1, -0.9], rng.standard_normal(bar_n)) * 0.04 * prog_frac
                L[seg] += riser + noise
                R[seg] += riser + noise
            # Pads: supersaw chords; open filter in the drop, swell in the build
            chord = prog[prog_i]
            pad = sum(_supersaw(_note(m), bar_n, rng=rng) for m in chord) / len(chord)
            cutoff = {"intro": 900, "build": 600 + 5000 * prog_frac ** 2, "drop": 7000,
                      "break": 700, "outro": 1500}[name]
            pad = _lowpass(pad, cutoff) * (0.22 if full else 0.10)
            if full:   # rhythmic stabs on the off-8ths
                gate = np.zeros(bar_n)
                for k in range(8):
                    if k % 2 == 1 or k in (0, 6):
                        gate[k * beat_n // 2:k * beat_n // 2 + int(beat_n * 0.4)] = 1
                pad = pad * (0.35 + 0.65 * _lowpass(gate, 60))
            L[seg] += pad * side[seg]
            R[seg] += _lowpass(np.roll(pad, 300), cutoff) * side[seg]
            # Bass: off-beat house bass in drop/outro, sustained root elsewhere
            root = _note(bass[prog_i])
            if full or name == "outro":
                for k in range(4):
                    bo = o + k * beat_n + beat_n // 2
                    n = beat_n // 2
                    tone = (_saw(root, n) * 0.5 + np.sin(2 * np.pi * root * np.arange(n) / SR))
                    tone = _lowpass(tone, 400) * _adsr(n, 0.005, 0.08, 0.6, 0.05)
                    L[bo:bo + n] += tone * 0.55
                    R[bo:bo + n] += tone * 0.55
            elif name != "intro" or b >= bars // 2:
                tone = np.sin(2 * np.pi * root * np.arange(bar_n) / SR) * 0.08
                L[seg] += tone * side[seg]
                R[seg] += tone * side[seg]
            # Hook: plucked supersaw lead with ping-pong delay (drop, and filtered in build)
            if full or (name == "build" and prog_frac > 0.5) or name == "intro":
                shift = 0
                half = (gbar % 2) * 16
                lead = np.zeros(bar_n)
                for k in range(16):
                    m = hook[half + k]
                    if m is None:
                        continue
                    n = six * 2
                    tone = _supersaw(_note(m + shift), n, voices=5, spread=0.18, rng=rng)
                    env = _adsr(n, 0.002, 0.12, 0.25, 0.08, 0.5)
                    so = k * six
                    lead[so:so + n] += (tone * env)[:max(0, min(n, bar_n - so))]
                lc = 6000 if full else (1200 if name == "intro" else 900 + 3000 * prog_frac)
                lead = _lowpass(lead, lc) * (0.30 if full else 0.16)
                d = int(0.75 * beat_n)
                L[seg] += lead
                R[seg] += lead * 0.8
                L[o + d:o + d + bar_n] += lead * 0.35
                R[o + 2 * d:o + 2 * d + bar_n] += lead * 0.25
            if full and b == 0:
                c = _crash(3 * SR, rng)
                L[o:o + len(c)] += c
                R[o:o + len(c)] += np.roll(c, 200)
            pos += bar_n
            gbar += 1
    mix = np.stack([L[:total], R[:total]], 1)
    mix /= np.abs(mix).max() + 1e-9
    mix = np.tanh(mix * 1.6) / np.tanh(1.6) * 0.9
    return mix.astype(np.float32), starts


def soundtrack(plan):
    """The audio for a plan: plan["music"] names a synth ("rave", "track") or an
    arrangement in crabshow.arrange.PIECES. Returns (stereo float32, label)."""
    name = plan.get("music", "rave")
    from . import arrange

    if name in arrange.PIECES:
        dur = sum(b for _, b in plan["sections"]) * 4 * 60.0 / plan["bpm"]
        kw = {}
        if name in ("halloween_v3", "halloween_v4"):
            kw = dict(prompt=plan.get("prompt"), t0=plan.get("prompt_t0", 0.3),
                      cps=plan.get("prompt_cps", 22.0), enter_s=plan.get("prompt_enter", 3.75))
        return arrange.PIECES[name](bpm=plan["bpm"], seconds=dur, **kw), name
    synth = synth_rave if name == "rave" else synth_track
    audio, _ = synth([tuple(s) for s in plan["sections"]], bpm=plan["bpm"])
    return audio, name
