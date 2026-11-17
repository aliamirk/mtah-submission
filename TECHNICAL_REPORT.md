# Los Sonámbulos: frame-level ball possession, FC Barcelona vs Athletic Club

More than a Hack 2027, technical document. Team: Los Sonámbulos. Code: <https://github.com/aliamirk/mtah-submission>

## 1. Summary

We label each of the 199,319 tactical-video frames as H, A, D or X with a rule-based pipeline built in four parts: nearest-player evidence for who has the ball, a Viterbi decoder that turns that evidence into possessions, a dead-ball model (fixed non-play ranges, capped Smart Tag windows and sustained low player speed), and a short restart prior from set-piece tags. It runs from the raw tracking and tag files in about 90 seconds and reproduces the submitted file byte for byte from a clean clone.

On the public 5-minute window (17 possessions) it scores spellF1 0.438 and macroF1 0.774. That window cannot rank versions: a bootstrap over its 17 possessions gives intervals of about +/-0.13, and the organisers' own baselines change order between the public window and the full match. We therefore chose settings from structure that can be checked over the whole match (possession counts and durations, dead-ball share, longest possessions) and from our own video checks, not from the public score.

## 2. Approach

### 2.1 Who has the ball: nearest-player evidence

For every frame with a tracked ball, the team of the tracked player closest to the ball (positions scaled to metres on a 105.3 x 68.0 m pitch) votes +1 for home or -1 for away; frames without a ball vote 0. On live frames in the public window this call is right 78% of the time overall, 91% when a player is within 1 m of the ball, and only 57% when the nearest home and away players are within 0.5 m of each other.

### 2.2 From votes to possessions: Viterbi decoding

Per-frame votes flicker, and the ranking metric counts each possession once, so we decode instead of voting over a window. Within each stretch of live play we run a two-state (home, away) Viterbi decode: each frame whose vote disagrees with the state costs 1, a switch costs 30 (switch\_penalty), and frames with no vote cost nothing. A switch therefore needs about 30 frames of net disagreement. At comparable possession counts this gave sharper handover timing than a rolling majority vote (transition F1 at 25 frames of about 0.22 against 0.08 to 0.20 for the window votes on the public window).

### 2.3 Dead ball (D)

A frame is D if any of three rules applies:

1. Fixed non-play ranges: before the first kickoff, from halftime through the frame before the second-half kickoff (so the 2,934 graded frames before that kickoff are D), and after the final graded frame. These ranges come from DATA.md and FAQ.md.
2. Set-piece tag windows: the 59 Smart Tagging set pieces mask their first 150 frames plus a 5-frame buffer. The tags' own windows are fixed 35 s spans (53 of 59), far longer than a stoppage, so we cap them. The two kickoff-start tags are unmasked from the published kickoff frames.
3. Sustained low player speed: where the 2 s rolling mean of the average reported player speed stays below 1.4 m/s for at least 12 s (gaps up to 3 s bridged), the run is dead. This rule matters most. It removes stoppages that carry no tag (fouls, goals and celebrations, hydration breaks, VAR checks, injuries, substitutions), which otherwise merge neighbouring possessions into spells of two to four minutes.

### 2.4 Ball gaps (X)

The ball is untracked on about 41% of graded frames, in long blocks (110 of 151 gaps last 5 s or more, about 19 s on average). Where the ball has been missing for 25 frames or more outside D, we output X and give the decoder no evidence. The scorer forward-fills X with the last label, so a gap inside a possession costs nothing and the decoder keeps the possession alive.

### 2.5 Restart prior

When a named-team corner, goal-kick or free-kick tag overlaps a quiet run (5 s slack), the tagged team gets a short prior (50 frames) after the run ends. On 32 hand-noted stoppages this raised the correct restarting team from 17 of 25 to 18 of 25: a small effect, kept because it is rule-based and cheap.

## 3. Assets used and dropped

**Used.** The tactical tracking (player positions, team of each track slot, per-player speed, ball position); the Smart Tagging set-piece tags (type, team, window); the published graded ranges and kickoff frames; the public 5-minute ground truth to check and compare versions (never read by the code that produces labels; only the evaluation step reads it).

**Used by hand, for validation only.** We watched the tactical video to check the dead-ball rule. All 12 quiet runs of 45 s or more and all 32 runs of 20 to 45 s contained a real stoppage, and 14 of 16 runs of 12 to 20 s did. The runs came from our own detector, so these checks measure precision, not recall. The 1.4 m/s threshold was chosen by eye from the four stoppages labelled in the public window and then checked in the video; the minimum run length (45 s, then 20 s, then 12 s) followed those video checks. The notes are not read by the pipeline.

**Tried and dropped** (each measured, with notes in the repository):

- Smart Tag phase votes: the phase tags come in exact mirrored pairs (attacker and defender), so uninverted they cancel and inverted they count each phase twice. Switching them off, together with narrowing the vote window, lifted the public-window spellF1 from 0.15 to 0.34.
- Reliability-weighted (soft) evidence in the decoder: no benefit at matched possession counts.
- Touch detection from ball velocity changes and from Kalman-filter innovations: handovers and ordinary passes were not separable (AUC 0.62 to 0.70 on 12 handovers).
- Reachability (nearest player to the extrapolated ball position), as a whole-window feature, around handovers and speed-gated: a better per-frame call (79.5% against 77.8%) but worse once decoded, at two horizons.
- Gating votes by distance to the nearest player: no gain at matched counts.
- Broadcast tracking: it aligns with the tactical track (offsets of 4014 and 26917 frames, median ball gap 1.07 m) but fills only 4.8% of the missing tactical ball frames, because both cameras lose the ball at the same moments.
- Placing handovers at the middle of ball gaps: only 2 of 172 switches fall inside a gap.

**Not used.** The broadcast video and homography for producing labels (the video was used for human inspection and overlay checks only).

## 4. Possession rules implemented

- A possession is a maximal run of H or A; D ends a possession and the next one starts after it.
- The team in possession changes only when evidence for the other team outweighs the switch cost, so loose balls, deflections and contested balls stay with the last team that had clear control.
- Ball gaps do not end a possession on their own.
- After a stoppage the new possession takes the team the evidence favours, with the tag prior for named restarts.
- There is no minimum duration, but the switch cost removes spells shorter than about a second.

## 5. Results and sensitivity

Public window: 15 predicted possessions against 17 true, mean 16.6 s against 14.2 s, spellF1 0.438, macroF1 0.774, D recall 0.67 and precision 0.78. The macroF1 is partly circular, because the speed threshold was chosen on the same four stoppages.

Whole match: 256 possessions, mean 14.5 s, 41.6% of graded frames dead (the public window has about 20%). In the 12 s configuration no possession was longer than two minutes. Before the dead-ball rules, dead ball was 8.6% of graded frames and the longest predicted possession was 239 s.

One-step perturbations of the final configuration:

| variant | possessions | mean length (s) | dead share | frames changed vs final | public spellF1 |
| --- | --- | --- | --- | --- | --- |
| final | 256 | 14.5 | 41.6% | 0.0% | 0.438 |
| no tag windows (cap 1 frame) | 260 | 14.6 | 40.2% | 1.5% | 0.438 |
| switch penalty 25 | 288 | 12.9 | 41.6% | 1.7% | 0.435 |
| switch penalty 38 | 227 | 16.3 | 41.6% | 1.3% | 0.477 |
| speed threshold 1.2 | 263 | 16.1 | 33.4% | 9.0% | 0.448 |
| speed threshold 1.6 | 252 | 13.4 | 46.8% | 5.5% | 0.425 |
| minimum run 8 s | 262 | 13.7 | 43.7% | 2.1% | 0.425 |
| minimum run 20 s | 254 | 15.7 | 37.2% | 5.1% | 0.438 |

Possession counts stay within about 12% of the final under every change. The dead-ball share is the most sensitive quantity (33% to 47% across the speed threshold). The public spellF1 moves between 0.425 and 0.477 under harmless changes, which is the noise level of 17 possessions.

## 6. Limitations

- Short possessions are not recovered. Six of the 17 public possessions last under 3.6 s; nearest-player agreement inside them is 49%, and touch, Kalman and reachability evidence gave no usable signal.
- Boundaries are loose. Around a handover the nearest-player call is close to a coin flip (49% to 57% agreement within a second), so long possessions start and end about 3.4 s off on the public window.
- Dead-ball recall is unmeasured. We have no labels outside the public window, and the speed rule does not mark stoppages shorter than 12 s. We also cannot tell whether the labels count player stillness around a stoppage or only the seconds the ball is out.
- The public window may not be typical, and we do not know the hidden-match score. The published baselines drop 0.08 to 0.2 between the public window and the full match, and we expect a drop too.
- All settings are tuned for this one match.

With more time we would fit a three-state (home, away, dead) hidden Markov model for dead ball with emissions estimated from hand-labelled stoppages in place of the fixed threshold, label two or three continuous stretches of the match to measure dead-ball recall, and check whether a video ball detector could recover the ball where the tracker loses it.

## 7. Reproduction

Environment: Python 3.13.1 with the pinned versions in requirements.txt (kloppy 3.18.0, numpy 2.5.4, pandas 3.1.0, PyYAML 6.0.3).

1. Place the data bundle in data/ (data/tactical/BAR-ATH\_tactical\_atd.txt, \_atd.xml and \_smart\_tagging.xml are the files the pipeline reads). The data is not in the repository.
2. Run: python main.py
3. The file is written to submission.csv (199,319 rows). Its SHA-1 from a clean clone is 1e8503203b232bf281f0a57925a315955b41b47f.

The run takes about 90 seconds. The speed signal behind the quiet-run rule is rebuilt from the raw tracking on first use (cache/ is not tracked). Experiments are in experiments/, and evaluation/run\_evaluation.py scores any file against the public ground truth.
