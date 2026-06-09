# Approximate FIFA's Annex C third-place allocation with constraint-satisfying matching

FIFA's regulations (Annex C, the "FIFA Matrix") define an exact mapping for all 495 combinations of which eight third-placed teams qualify, assigning each to a specific Round of 32 slot. Transcribing 495 rows from the regulations PDF is impractical and error-prone. We instead assign the eight qualified thirds to the eight slots by deterministic constraint-satisfying matching against the verified per-slot eligibility lists in `bracket.json` (e.g. match 74 accepts thirds from groups A/B/C/D/F only; a group winner never meets the third from its own group). Any valid matching preserves the eligibility constraints; the difference from FIFA's exact table only shuffles *which* eligible third meets which winner — a second-order effect on stage probabilities. This is the standard approximation in public tournament models.

## Consequences

- Simulated R32 pairings can differ from the real bracket in third-place slots even when group outcomes are predicted exactly.
- If exact pairing fidelity ever matters (e.g. match-level betting), Annex C must be transcribed and this module replaced.
