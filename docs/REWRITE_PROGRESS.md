# S-AI Rewrite Progress Tracker

Updated continuously during autonomous A-Z rewrite session.

## Baseline (2026-05-24)

- Git: clean.
- Last commit before session: `bc64458` (style guide draft).
- mdBook build: OK, no KaTeX errors across all chapters.
- Total source lines: 11,939 across 30 `.md` files.

## Phase Status

| Phase | Description | Status | Notes |
|---|---|---|---|
| 0 | Baseline + tracker | done | This file. |
| 1 | STYLE_GUIDE.md professional rewrite | pending | |
| 2 | SUMMARY/index/book.toml/README sync | pending | |
| 3 | Part 0 + Part I (Ch0-Ch3) | pending | Ch0 new file. |
| 4 | Part II ASR (Ch4-Ch7) | pending | |
| 5 | Part III TTS+codecs (Ch8-Ch10) | pending | |
| 6 | Part IV Speech LLM (Ch11-Ch13) | pending | |
| 7 | Part V app layer (Ch14, Ch15, Ch21) | pending | Ch15 + Ch21 already expanded, audit tone. |
| 8 | Part VI Vietnamese (Ch16-Ch17 + App E) | pending | Ch16 already has industry section. |
| 9 | Part VII production (Ch18-Ch20 + App F) | pending | Ch19/Ch20 already expanded, audit tone. |
| 10 | Appendices A-D + references.md | pending | |
| 11 | Repo docs + theme polish | pending | |
| 12 | Final QA + SESSION_REPORT | pending | |

## File-level tracking

Legend: `pending` / `in-progress` / `done` / `blocked`. Status `audit` means already expanded in previous session but needs tone audit under the corrected style guide.

### Documentation

| File | Lines | Status | Last commit |
|---|---|---|---|
| `docs/STYLE_GUIDE.md` | 384 | pending (rewrite) | bc64458 |
| `docs/research-q4-2025-models.md` | 162 | pending (polish) | (earlier) |
| `docs/AUTONOMOUS_SESSION_REPORT.md` | 188 | pending (review) | (earlier) |
| `README.md` | 62 | pending | (earlier) |

### Root / Theme / Config

| File | Status | Notes |
|---|---|---|
| `book.toml` | pending | Metadata, math config review. |
| `theme/custom.css` | pending | Typography, callouts, tables. |
| `theme/custom.js` | pending | Minor utilities. |
| `theme/katex-macros.txt` | pending | KaTeX macros. |
| `.github/workflows/publish-mdbook.yml` | pending | Already minimal. |
| `.gitignore` | pending | Verify `/book/`. |

### Book source

| File | Lines | Status | Notes |
|---|---|---|---|
| `src/index.md` | 257 | pending | Sync Phần 0. |
| `src/SUMMARY.md` | 56 | pending | Add Chương 0. |
| `src/references.md` | 2 | pending | Placeholder. |
| `src/part0/00-foundations-review.md` | (new) | pending | Create from scratch. |
| `src/part1/01-nlp-to-speech.md` | 938 | audit | Already comprehensive. |
| `src/part1/02-audio-fundamentals.md` | 512 | pending | Expand DSP. |
| `src/part1/03-speech-representations.md` | 321 | pending | Expand SSL. |
| `src/part2/04-asr-foundations.md` | 549 | pending | |
| `src/part2/05-modern-asr.md` | 313 | pending | |
| `src/part2/06-whisper-canary.md` | 368 | pending | |
| `src/part2/07-streaming-asr.md` | 270 | pending | |
| `src/part3/08-tts-foundations.md` | 497 | pending | |
| `src/part3/09-end-to-end-tts.md` | 429 | pending | |
| `src/part3/10-audio-codecs.md` | 441 | pending | |
| `src/part4/11-speech-llms.md` | 520 | audit | Already has Q4/2026 update. |
| `src/part4/12-multimodal-omni.md` | 232 | pending | Expand. |
| `src/part4/13-full-duplex.md` | 195 | pending | Expand. |
| `src/part5/14-speech-classification.md` | 284 | pending | Expand. |
| `src/part5/15-speech-translation.md` | 997 | audit | Already comprehensive. |
| `src/part6/16-vietnamese-speech.md` | 476 | audit | Already has industry section. |
| `src/part6/17-vietnamese-datasets.md` | 231 | pending | Expand. |
| `src/part7/18-training-frameworks.md` | 482 | pending | Expand frameworks. |
| `src/part7/19-inference-engines.md` | 1124 | audit | Already comprehensive. |
| `src/part7/20-production-systems.md` | 777 | audit | Already comprehensive. |
| `src/part7/21-wake-word-detection.md` | 753 | audit | Already comprehensive. |
| `src/appendices/appendix-a-notation.md` | 113 | pending | |
| `src/appendices/appendix-b-proofs.md` | 323 | pending | |
| `src/appendices/appendix-c-mapping.md` | 115 | pending | Expand mapping. |
| `src/appendices/appendix-d-code.md` | 152 | pending | Sync chapter numbers. |
| `src/appendices/appendix-e-vietnamese-resources.md` | 94 | pending | Expand resources. |
| `src/appendices/appendix-f-tool-comparison.md` | 118 | pending | Expand matrices. |

## Stop conditions log

Empty (no blockers yet).

## Build log

| Timestamp | Trigger | Result | KaTeX errors |
|---|---|---|---|
| 2026-05-24 01:21 | Baseline | OK | 0 |
