# Frame-level ball possession (More than a Hack 2027)

Pipeline: nearest-player evidence, Viterbi decoding, dead-ball rules (fixed ranges, capped Smart Tag
windows, sustained low player speed), short restart prior. See the technical document.

## Reproduce
1. Python 3.13.1, then `pip install -r requirements.txt`.
2. Put the data bundle in `data/` (not included). The pipeline reads `data/tactical/BAR-ATH_tactical_atd.txt`,
   `_atd.xml` and `_smart_tagging.xml`.
3. `python main.py` writes `submission.csv` (199,319 rows, about 90 seconds).
4. `python evaluation/validate_submission.py submission.csv` checks the format.

SHA-1 of the submitted file: `1e8503203b232bf281f0a57925a315955b41b47f`.
`experiments/` holds the analysis scripts used during development. No match data is included.

Development log, experiment register and file map: see NOTES.md.
