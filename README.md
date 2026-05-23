# S-AI — Speech AI

**Speech AI: From Signal Processing to Full-Duplex Dialogue**
Sách học thuật về Speech AI dành cho data scientist NLP/LLM, viết bằng tiếng Việt.

📖 **Đọc online**: <https://tuandung222.github.io/S-AI/>

## Stack

- **Static site generator**: [mdBook](https://rust-lang.github.io/mdBook/) — Rust-based, fast, academic-grade
- **Math rendering**: [mdbook-katex](https://github.com/lzanini/mdbook-katex) preprocessor (build-time, không phụ thuộc runtime MathJax)
- **Typography**: Source Serif 4 (body) + Inter (heading) + JetBrains Mono (code)
- **Hosting**: GitHub Pages (gh-pages branch)
- **CI/CD**: GitHub Actions

## Cấu trúc

```
src/                  # mdBook markdown source
├── SUMMARY.md       # Navigation (auto-generated)
├── index.md         # Trang chủ
├── part1/...         # Phần I — Nền tảng
├── part2/...         # Phần II — ASR
├── part3/...         # Phần III — TTS
├── part4/...         # Phần IV — Speech LLMs & Multimodal
├── part5/...         # Phần V — Understanding & Translation
├── part6/...         # Phần VI — Tiếng Việt
├── part7/...         # Phần VII — Tools & Production
├── appendices/...    # Phụ lục A-F
└── references.md     # Tài liệu tham khảo

book.toml             # mdBook config
theme/                # Custom CSS, JS, KaTeX macros
.github/workflows/    # GitHub Actions deploy
docs/                 # STYLE_GUIDE.md and writing conventions
```

## Build locally

```bash
# Cài mdBook + preprocessor (cần Rust toolchain)
cargo install mdbook --version 0.4.40
cargo install mdbook-katex --version 0.9.4

# Build
mdbook build      # Output ở book/

# Live preview
mdbook serve --open    # localhost:3000
```

## Lịch sử

Repo này gốc fork từ `dung-vpt/speech-processing-book` (Quarto book).
Đã migrate qua 3 stacks: Quarto → MkDocs Material → mdBook (final).

Branch `quarto-legacy` lưu snapshot Quarto state.

## License

MIT — xem [LICENSE](LICENSE).
