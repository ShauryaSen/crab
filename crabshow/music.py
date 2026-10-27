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
