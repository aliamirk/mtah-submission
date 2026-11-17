# Development notes

Working log for the possession pipeline: what the metric rewards, what we learned about the data, every idea we tested with its result, and the open questions. The submitted pipeline is described in the technical document. This file is the evidence behind it.

All scores are on the public 5-minute window (17 possessions) unless stated. That window cannot rank versions: a bootstrap over its possessions gives intervals of about +/-0.13 for large changes, and one missed possession moves spellF1 by roughly 0.06. We used it to check that the pipeline is correct and to see mechanisms. Settings were chosen from whole-match structure (possession counts and durations, dead-ball share, longest possessions) and from video checks.

## 1. Lessons about the metric

Measured by damaging the public ground truth and scoring the result (scorer: `evaluation/possession_metrics.py`).

- A short wrong blip inside a possession splits it into pieces and is very expensive. Ten frames of the other team in the middle of the longest possession: macroF1 0.999, spellF1 0.911, possessions 17 to 19.
- Timing errors of 1 to 3 frames are free. 8 frames late costs about 5 points (spellF1 0.953). A full second late costs about 22 points (0.777).
- X inside a possession is free, because the scorer forward-fills it. X at a handover costs more the longer it lasts, and the cost falls on short possessions: 25 frames of X at the start of a 19-frame possession cost about 9 points.
- Merging two neighbouring possessions costs about 7.5 points (0.925), and merging also fuses same-team neighbours.
- Real possessions range from under 1 second to over 40 seconds (public window: median 7.6 s, mean 14.2 s, 6 of 17 shorter than 3.6 s), so any minimum-duration rule has a price.
- A system that finds only the 11 long possessions cannot exceed 2 x 11 / (11 + 17) = 0.786 at one IoU threshold, even if perfect.

## 2. Facts about the data

- ATD `frame_id` is the video frame plus 1. Tracking projected through the homography lands on the players and the ball in the tactical video, so tracking, labels and video share one clock.
- The ball is tracked on 56.9% of frames and missing on 40.7% of graded frames, in long blocks. In our output there are 151 gap runs of 1 s or longer, 110 of them 5 s or longer, about 19 s on average.
- Nearest-player team call on live frames in the public window: 78% overall. By distance from the ball to the nearest player: within 1 m 91%, 1 to 2 m 83%, 2 to 3 m 72%, 3 to 5 m 60%. By the gap between the nearest home and nearest away player: under 0.5 m 57%, 2 to 3 m 74%, 5 to 8 m 89%, over 12 m 100%.
- Inside the 6 shortest public possessions the nearest-player call agrees with the label on 49% of frames (long possessions: 74%).
- Smart Tagging: `Team` is N/A for all 26 throw-ins and the 6 kickoffs. 53 of the 59 set-piece tags are exactly 35 s windows; kickoff tags are 5 s. Phase tags (7.7 to 83 s) come in exact mirrored attacker/defender pairs: build up with defending in attacking third 22 of 22, progression with defending in middle third 66 of 66, attacking transition with defensive transition 24 of 24, creating chances with defending in defensive third 28 of 30.
- Set-piece windows do not pin the restart: in the two public examples the restart fell 12.2 s and 2.8 s into the window, and the stoppage began 6 to 12 s before the tag. On 32 hand-noted stoppages the named team on a same-kind tag was the restarting team in 9 of 11 cases.
- Dead ball is 19.6% of the public window (4 stoppages). The possession rate is about 4.2 per minute of live play in the window and in the whole-match output (258 possessions in about 62 minutes of live play).

## 3. Pipeline decisions

| component | choice | config key |
|---|---|---|
| evidence | team of the player nearest the ball, metres, hard +1/-1 | |
| decoder | two-state Viterbi per live stretch, switch cost 30 | `switch_penalty`, `soft_evidence: false` |
| ball gaps | X when the ball has been missing 25+ frames, no evidence given | `max_carry_gap` |
| fixed dead ranges | before first kickoff, halftime through the frame before the second-half kickoff, after the last graded frame | `nonplay_regions` |
| tag windows | set-piece tags capped at 150 frames plus a 5-frame buffer; kickoff tags unmasked from the kickoff frame | `dead_ball_max_duration_frames` |
| quiet runs | 2 s mean of the mean player speed below 1.4 m/s for 12 s or more, gaps up to 3 s bridged | `quiet_*` |
| restart prior | named-team corner, goal-kick or free-kick tag overlapping a quiet run (5 s slack) gives a 50-frame prior after the run | `restart_from_runs`, `restart_prior_*` |
| switched off | phase-tag votes, reachability gate | `tag_vote_weight: 0`, `reach_enabled: false` |

## 4. Experiment register

| # | idea | result | decision |
|---|---|---|---|
| 1 | Window majority vote (organisers' baselines) | public spellF1 0.04 (raw), 0.29 (+/-25), 0.33 (+/-50), 0.42 (+/-125). Full match (organisers): 0.05, 0.20, 0.24, 0.22, so the public ranking reverses | not used |
| 2 | First pipeline: phase-tag votes (weight 3) with a 150-frame window | spellF1 0.148, macroF1 0.446, H/A recall 0.50/0.32, below the plain baseline | dropped |
| 3 | Tag votes off, window 50 | 0.337 / 0.658, recall 0.80/0.64 | tag votes dropped |
| 4 | Tag dead mask uncapped against capped at 150 frames | uncapped: D recall 0.46 but 528 live frames marked D and two possessions erased. Capped: D recall 0.16, spellF1 0.410 | cap kept as a coarse hint; stoppages now come mainly from quiet runs |
| 5 | Viterbi decoder in place of the window vote | transF1@25 0.17 to 0.21-0.22. Whole-match possessions fall as the penalty grows (288, 256, 227 at 25, 30, 38); public spellF1 flat across 25 to 40 | adopted, penalty 30 |
| 6 | Soft (reliability-weighted) evidence | public spellF1 0.490, but the matched-count control (hard decoder, 15 possessions: 0.450) had transF1 0.222 against 0.171 and higher macroF1; the gain came from fewer spells | dropped |
| 7 | Touch detection (velocity kinks, Kalman innovations) | AUC separating 12 handovers from 655 control windows: 0.70 (kink), 0.62 to 0.65 (Kalman); right team at the kink 5 of 9 at handovers | dropped |
| 8 | Reachability (nearest player to the extrapolated ball position) | whole window 78.5% against 77.8%; only balls at 10 m/s or more differ (+6.0 points at 0.5 s, 90% interval 2.3 to 10.1; +8.5, -3.3 to +16.3 at 1.0 s). Integrated: team accuracy 0.864 and 0.855 against 0.880, net labels -88 and -137 | dropped |
| 9 | Gate votes by distance to the nearest player | no consistent gain at matched possession counts | dropped |
| 10 | Broadcast tracking as a second ball source | constant offsets of 4014 and 26917 frames align the ball tracks (median 1.07 m, 305 of 658 ten-second chunks verified), but only 4.8% of the missing tactical ball frames are recovered | dropped |
| 11 | Quiet runs as dead ball | In video: 12 of 12 runs of 45 s or more, 32 of 32 runs of 20 to 45 s and 14 of 16 runs of 12 to 20 s contained a stoppage (precision of detector-flagged runs; recall unknown). Dead share 8.6% to 24% (45 s) to 37% (20 s) to 41.6% (12 s). Longest possession 239 s to 84 s | adopted at 12 s |
| 12 | Restart prior from named set-piece tags near a quiet run | correct restarting team 17 to 18 of 25; the previous team keeps the ball in only 10 of 24 | kept (small) |
| 13 | Midpoint placement of handovers inside ball gaps | only 2 of 172 switches sit inside a gap; simulated gain 0.024 came from 1 to 2 gaps per seed | dropped |
| 14 | Kickoff-start tags as dead ball | the 5 s window contains the published kickoff frame; masking it labelled 43 live frames dead | unmasked from the kickoff frame |
| 15 | Second-half frames before kickoff (110,808 to 113,741) | graded and all dead; the first version labelled almost all of them H or A | forced D through `nonplay_regions` |
| 16 | Remove the tag windows under the final decoder | D recall 0.67 to 0.63, precision 0.78 to 0.82, 1.5% of frames differ | kept, by the rule fixed before the run |

### Sensitivity of the final configuration

| variant | possessions | mean length (s) | dead share | frames changed | public spellF1 |
|---|---|---|---|---|---|
| final | 256 | 14.5 | 41.6% | 0.0% | 0.438 |
| no tag windows (cap 1 frame) | 260 | 14.6 | 40.2% | 1.5% | 0.438 |
| switch penalty 25 | 288 | 12.9 | 41.6% | 1.7% | 0.435 |
| switch penalty 38 | 227 | 16.3 | 41.6% | 1.3% | 0.477 |
| speed threshold 1.2 | 263 | 16.1 | 33.4% | 9.0% | 0.448 |
| speed threshold 1.6 | 252 | 13.4 | 46.8% | 5.5% | 0.425 |
| minimum run 8 s | 262 | 13.7 | 43.7% | 2.1% | 0.425 |
| minimum run 20 s | 254 | 15.7 | 37.2% | 5.1% | 0.438 |

Possession counts stay within about 12% of the final under every change. The dead share is the sensitive quantity. Public spellF1 moves between 0.425 and 0.477 under harmless changes, which is the noise level of 17 possessions.

## 5. Dead-ball validation (hand notes, kept outside the repository)

We watched the tactical video to check the speed rule: 60 detector-flagged runs (12 + 32 + 16 by length band, see experiment 11), plus an earlier set of 16 short quiet runs of 4 to 21 s (5 stoppages, 3 live with a stoppage at the edge, 8 plain live play) and 13 ball gaps of 5 s or more (5 dead, 4 mixed, 4 live). The notes are not read by the pipeline. They influenced one parameter, the minimum run length (45 s, then 20 s, then 12 s). The 1.4 m/s threshold was chosen by eye from the four stoppages labelled in the public window, so the public macroF1 of 0.774 is partly circular.

## 6. Open questions

- What the labels count as dead ball. The speed rule reproduces the four public stoppages within a couple of seconds, yet in the video the visible stoppage was sometimes shorter than the labelled stretch or absent inside it. We could not settle whether the labels follow player stillness or the seconds the ball is out.
- Short possessions. None of nearest-player agreement, touch detection, Kalman innovations or reachability gave usable evidence for the 6 public possessions under 3.6 s.
- Dead-ball recall outside the public window is unmeasured. A few continuous labelled stretches of the match would measure it.
- The dead share moves between 33% and 47% with the speed threshold.
- The public window may not be typical of the match.

## 7. Reproduction

```
pip install -r requirements.txt        # Python 3.13.1
python main.py                          # writes submission.csv, about 90 seconds
python evaluation/validate_submission.py submission.csv
experiments/clean_clone_check.sh        # clones the repo, reruns, compares checksums
```

SHA-1 of the submitted file: `1e8503203b232bf281f0a57925a315955b41b47f`. The speed signal behind the quiet-run rule is rebuilt from the raw tracking on first use (`cache/`, not tracked).

## 8. File map

- `main.py`: pipeline entry point. `config.yaml`: every parameter. `games.py`: organisers' per-match constants.
- `src/`: `load_data.py` (tracking to frame table), `preprocess.py` (ball kinematics, nearest-player team, margin, reach gate), `smart_tags.py` (dead-ball mask and tag priors), `quiet.py` (player speed and quiet runs), `restart.py` (restart prior), `decoder.py` (Viterbi), `decoder_assigner.py` (labels), `evaluate.py` and `output.py` (local scoring and file writing), `config_utils.py`.
- `evaluation/`: organisers' scorer and the public ground truth. `starter/`: organisers' loaders.
- `experiments/`: analysis scripts. Most read caches that `main.py` or the script itself rebuilds from the raw data.
  - scoring and comparison: `paired_compare.py`, `changed_labels.py`, `team_acc.py`, `boundary_errors.py`, `spell_table.py`, `spell_breakdown.py`, `spell_diff.py`
  - whole-match checks: `whole_match_diag.py`, `long_spell_gaps.py`, `gap_switches.py`, `gap_simulation.py`, `sensitivity.sh`, `sensitivity_report.py`
  - evidence tests: `raw_signal_per_spell.py`, `evidence_calibration.py`, `touch_test.py`, `kalman_touch.py`, `reach_by_speed.py`, `reach_flip_windows.py`, `reach_perframe_check.py`, `gate_sweep.py`
  - decoder sweeps: `decoder_sweep.py`, `fast_sweep.py`
  - tags and dead ball: `tag_durations.py`, `tag_mirrors.py`, `setpiece_vs_gt.py`, `ball_gaps.py`, `quiet_in_long_spells.py`, `quiet_batch.py`, `more_quiet_runs.py`, `restart_check.py`
  - broadcast: `broadcast_sync.py`
  - helpers: `run_exp.sh`, `clean_clone_check.sh`

## 9. Rules and sources

- Every submitted label comes from `main.py`. The public ground truth is read only by the evaluation step. The private ground truth was never used. The match data is not in the repository.
- Hand-written notes were used to validate and to choose the minimum run length, never to write labels.
- Ideas from outside: time-to-intercept pitch control (reachability) comes from the sports-analytics literature, and aligning the broadcast tracking came up in a public participant repository. Both were implemented independently, and no code was copied.
