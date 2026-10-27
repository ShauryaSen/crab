"""Prompt -> choreography plan, without an LLM.

Reads a sentence like "light up, make a heart, dance on the drop, then spell HI" and
lays the steps out on the 20-bar track (intro 4 / build 4 / drop 8 / outro 4), in the
order they were written. Walks get enough bars for the distance; the first dance or jump
after the build lands on the drop. Anything it does not recognise is ignored.

An LLM can write the same JSON for richer prompts; this keeps "type anything" honest
for the common words, and reply videos ("spell my name!") need nothing more.
"""

from __future__ import annotations

import re

SHAPES = {"heart": "heart", "crab": "crab", "spiral": "spiral", "circle": "circle",
          "ring": "circle", "grid": "grid", "square": "grid", "line": "rows", "lines": "rows"}
DANCES = {"brazil": "dance_brazilian", "samba": "dance_brazilian", "wings": "dance_dream_wings",
          "waist": "dance_waist", "maze": "dance_maze"}
GESTURES = {"wave": "gesture_hello", "hello": "gesture_hello", "hi!": "gesture_hello",
            "bow": "gesture_bow", "salute": "gesture_salute", "paw": "gesture_paw"}
PALETTES = ["love", "ice", "gold", "ocean", "crab", "rainbow", "white"]
COLOR_WORDS = {"red": "crab", "pink": "love", "blue": "ocean", "gold": "gold", "yellow": "gold",
               "white": "white", "rainbow": "rainbow", "ice": "ice", "orange": "crab"}


def parse(prompt: str) -> list[dict]:
    """Ordered list of steps: {"do": ..., ...}."""
    steps = []
    for clause in re.split(r",|\bthen\b|\band\b|;|\.", prompt):
        c = clause.strip()
        low = c.lower()
        if not c:
            continue
        pal = next((p for w, p in COLOR_WORDS.items() if re.search(rf"\b{w}\b", low)), None)
        m = re.search(r"\b(?:spell|write|say|form the word|make the word)\s+[\"']?([A-Za-z0-9!?]{1,6})", c)
        if m:
            steps.append({"do": "form", "text": m.group(1).upper(), "palette": pal})
            continue
        shape = next((s for w, s in SHAPES.items() if re.search(rf"\b{w}s?\b", low)), None)
        if shape and re.search(r"\b(make|form|become|shape|into|be|draw)\b", low):
            steps.append({"do": "form", "shape": shape, "palette": pal})
            continue
        if re.search(r"\blight(s)? up\b|\bglow\b|\bwake up\b", low):
            steps.append({"do": "light_up", "palette": pal or "rainbow"})
            continue
        if re.search(r"\bjump", low):
            steps.append({"do": "jump"})
            if "dance" not in low:
                continue
        if re.search(r"\bdanc|\brave\b|\bgo wild\b|\bparty\b", low):
            move = next((d for w, d in DANCES.items() if w in low), "dance_brazilian")
            steps.append({"do": "dance", "move": move})
            continue
        g = next((v for w, v in GESTURES.items() if w in low), None)
        if g:
            steps.append({"do": "gesture", "move": g})
            continue
        if re.search(r"\bbounce|\bbob\b|\bhead.?bang", low):
            steps.append({"do": "bounce"})
            continue
        if re.search(r"\bscuttle|\bsideways|\bwalk\b", low):
            steps.append({"do": "scuttle"})
    return steps


def to_plan(prompt: str, name: str, n: int = 72, style: str = "moonbeach", fx: bool = True):
    steps = parse(prompt)
    acts = []
    pal_cycle = iter(["love", "ice", "gold", "ocean", "crab"] * 3)
    bar = 0.5
    if not steps or steps[0]["do"] != "light_up":
        steps.insert(0, {"do": "light_up", "palette": "rainbow"})
    dropped = False
    for s in steps:
        d = s["do"]
        if d == "light_up":
            acts.append({"bar": bar, "do": "light_up", "palette": s.get("palette") or "rainbow",
                         "bars": 3})
            acts.append({"bar": 2, "do": "bounce", "bars": 2, "every": 2, "amp": 0.03})
            bar = 4
        elif d == "form":
            pal = s.get("palette") or next(pal_cycle)
            length = 3.8 if bar < 8 else 3.5
            a = {"bar": bar, "do": "form", "bars": length, "palette": pal,
                 "every": [1, 0.25] if bar + length <= 8.01 else 1, "amp": 0.033}
            if "text" in s:
                a["text"] = s["text"]
            else:
                a["shape"] = s["shape"]
                if s["shape"] == "heart":
                    a["filled"] = True
            acts.append(a)
            bar += length if bar + length > 8 or bar >= 8 else 0
            if bar < 8:
                bar = 8
        elif d in ("dance", "jump") and not dropped and bar <= 8:
            bar = 8
            acts.append({"bar": 8, "do": "jump", "bars": 1, "ripple": "center", "ripple_s": 0.6})
            dropped = True
            bar = 9
            if d == "dance":
                acts.append({"bar": 9, "do": "dance", "move": s["move"], "bars": 2.5,
                             "ripple": "center"})
                bar = 11.5
        elif d == "dance":
            acts.append({"bar": bar, "do": "dance", "move": s["move"], "bars": 2.5,
                         "ripple": "center"})
            bar += 2.5
        elif d == "jump":
            acts.append({"bar": bar, "do": "jump", "bars": 1, "ripple": "center", "ripple_s": 0.6})
            bar += 1
        elif d == "gesture":
            acts.append({"bar": bar, "do": "gesture", "move": s["move"], "bars": 2.5})
            bar += 2.5
        elif d == "bounce":
            acts.append({"bar": bar, "do": "bounce", "bars": 2, "every": 0.5, "amp": 0.03})
            bar += 2
        elif d == "scuttle":
            acts.append({"bar": bar, "do": "shift", "by": [1.8, 0], "bars": 1.5, "every": 0.5,
                         "amp": 0.03})
            acts.append({"bar": bar + 1.5, "do": "shift", "by": [-1.8, 0], "bars": 1.5,
                         "every": 0.5, "amp": 0.03})
            bar += 3
        if bar > 17.5:
            break
    if bar <= 15.5 and acts[-1]["do"] not in ("gesture",):
        acts.append({"bar": bar, "do": "gesture", "move": "gesture_hello", "bars": 2.5})
        bar += 2.5
    if not any(a["do"] == "jump" for a in acts):
        acts.append({"bar": 8, "do": "jump", "bars": 1, "ripple": "center", "ripple_s": 0.6})
    end = max(bar, 17.5)
    acts.append({"bar": min(end, 18.5), "do": "lights_out", "bars": 1.5})
    acts.sort(key=lambda a: a["bar"])
    plan = {
        "name": name, "prompt": prompt, "n": n, "style": style, "bpm": 128, "seed": 1,
        "sections": [["intro", 4], ["build", 4], ["drop", 8], ["outro", 4]],
        "acts": acts,
        "camera": [[0.0, [0, 0, 0.05], 1.0, 90, -80], [7.0, [0, 0, 0.05], 1.0, 90, -70],
                   [14.0, [0, 0, 0.05], 1.0, 90, -60], [15.3, [0, 0, 0.1], 0.75, 85, -35],
                   [17.5, [0, 0, 0.1], 0.45, 70, -25], [21.0, [0, 0, 0.1], 0.5, 115, -30],
                   [23.5, [0, 0, 0.05], 1.0, 90, -72], [37.5, [0, 0, 0.05], 1.05, 90, -80]],
        "prompt_hold": 6.5,
        "captions": [{"t0": 33.5, "t1": 37.5, "y": 0.74, "lines": [
            [f"{n} robot crabs. {n} physics sims.", 0.62],
            ["One prompt. Nothing keyframed.", 0.62],
            ["Reply with yours", 0.6]]}],
    }
    if fx:
        plan["fx"] = {"floor": {"from": 0, "tile": 0.5, "half": 9, "pattern": "ripple",
                                "energy": [[0, 0.35], [4, 0.35], [7.9, 0.9], [8, 1.0],
                                           [11.5, 0.9], [12, 0.35], [17, 0.6], [20, 0.0]]},
                      "trails": {"len_s": 0.8},
                      "lasers": [{"from": 8, "to": 12, "n": 8}],
                      "pillars": [{"at": 8, "beats": 3}],
                      "confetti": [{"at": 8, "n": 500}]}
    return plan
