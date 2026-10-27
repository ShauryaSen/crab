# Posting kit for round 1 (entries close Oct 30)

The rules ask for a video longer than 15 seconds on X or YouTube, the tags
`#Jumper #CrabRobot #OpenSourceRobot`, and a mention of **@KingKong Robotics**. Judging is 60%
audience response and 40% expert review. Sharing ideas, debugging and funny failures
counts in your favour.

## Main post (X)

> I typed one prompt and 72 robot crabs threw a rave 🦀🪩
>
> "light up, make a heart, dance on the beat drop, then spell HI"
>
> Every crab is its own physics sim running KingKong's open-source trained RL policies.
> Walking, jumping and dancing all come from the policies. Nothing is keyframed.
>
> Reply with a word and I'll make them spell it 👇
>
> Code: github.com/shauryasen/crab
> @KingKong Robotics #Jumper #CrabRobot #OpenSourceRobot

Attach the 4:5 video (`rave_v5_moonbeach.mp4`, or v4 if v5 is not ready).

## YouTube Shorts

Title: **72 robot crabs, one prompt: a physics-simulated crab rave 🦀**
Use the 9:16 file and the same description. Tag @KingKong Robotics and add the three hashtags.

## Follow-ups (one every day or two, to keep the audience score up)

1. **How it works** (`rave_v3_howitworks.mp4`): "Left: the render. Right: mission control.
   Each crab gets its spot from a Hungarian assignment, so no paths cross, and the closest
   any two crabs ever come is 0.67 m."
2. **Reply videos:** take the best reply words and run
   `scripts/plan_from_prompt.py "<their prompt>" <name>`. Render, then reply to that
   comment with the video.
3. **Bloopers:** crabs stuck on dunes, the garbled first renders, the invisible
   moonlight bug. The rules reward sharing setbacks.

## Checklist

- [ ] The video is longer than 15 s (ours are 37.5 s).
- [ ] The three hashtags and @KingKong Robotics are in the post.
- [ ] Link to the repo (the challenge is for open-source projects).
- [ ] Use the thumbnail `thumbnail.png` (YouTube).
- [ ] No branded skins or copyrighted music (ours is original).
