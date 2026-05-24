# S-AI Rewrite Progress Tracker

Updated during the 2026-05-25 autonomous full content review, rewrite, and verification pass.

## Current Baseline

- Repository: `/Users/admin/TuanDung/repos/speech-processing-book`.
- Book engine: mdBook with `mdbook-katex` and `mdbook-mermaid`.
- Public content scope: `src/index.md`, Chương 0-21, Phụ lục A-F, `src/references.md`.
- Validation status: `mdbook build` passes locally.

## Phase Status

| Phase | Description | Status | Notes |
|---|---|---|---|
| A | Baseline build and QA scan | done | Build passed; scan found stale `TBD`, tone issues, static figures, and Mermaid CI gap. |
| B | Build/deploy/style-guide consistency | done | CI now installs `mdbook-mermaid`; KaTeX underscore guidance corrected. |
| C | High-risk content rewrite | done | Ch16, Ch17, Ch20 rewritten for evidence discipline and lecturer tone. |
| D | Visualization upgrade | done | Core architecture figures converted from static placeholders to Mermaid diagrams. |
| E | Frontier/SOTA claim normalization | done | Ch12, Ch13, Ch14, Ch15, Ch20, Ch21 and appendices adjusted to avoid overclaiming. |
| F | Final verification and docs sync | in-progress | QA docs updated; final scan/build/commit/push pending. |

## File Groups Reviewed

| Group | Status | Notes |
|---|---|---|
| Root/build | done | `.github/workflows/publish-mdbook.yml` now installs `mdbook-mermaid`. |
| Style guide | done | `docs/STYLE_GUIDE.md` matches actual KaTeX safety rules. |
| ASR/TTS/Codec visuals | done | Ch4, Ch6, Ch8, Ch9, Ch10 diagrams converted to Mermaid where they were core architecture figures. |
| Speech LLM / Omni | done | Ch11 diagrams converted; Ch12 frontier claims tempered. |
| Applications | done | Ch14, Ch15, Ch21 claims/tone normalized; Ch15 frontier language rewritten. |
| Vietnamese content | done | Ch16/Ch17 benchmark and industry language rewritten more cautiously. |
| Production | done | Ch20 rewritten for professional production-system framing. |
| Appendices | done | Appendix C mapping figure converted; Appendix F SOTA wording normalized. |

## Final Scanner Targets

The final verification pass checks public `src/**/*.md` for:

- `TBD`, `TODO`, `FIXME`, `placeholder`, `coming soon`, `lorem`.
- Forbidden casual phrases such as `Bùm`, `đừng hoảng`, `Anyway`, `Nói thật`, `cực kỳ elegant`.
- Static figure placeholders matching `<img src="fig-`.
- mdBook build errors, KaTeX parse errors, and Mermaid preprocessor failures.

## Stop Conditions

No stop condition was triggered. No new chapter-scale restructuring was performed; changes stayed within the approved full review/rewrite/verify scope.
