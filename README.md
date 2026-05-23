# S-AI — Speech AI

**Speech AI: From Signal Processing to Full-Duplex Dialogue**

Sách học thuật bằng tiếng Việt về Speech AI cho data scientist nền tảng NLP/LLM và CV. Cuốn sách gồm 8 phần (bao gồm Phần 0 nền tảng cổ điển), 6 phụ lục, tổng cộng 22 chương (Chương 0 đến Chương 21), bao quát từ DSP nền tảng đến Speech LLM frontier và production deployment.

Đọc online: <https://tuandung222.github.io/S-AI/>.

## Audience

- Data scientist NLP/LLM (BERT, GPT, Transformer) muốn chuyển sang Speech AI.
- Data scientist CV cần modality âm thanh cho multimodal.
- ML/AI engineer xây dựng voice AI pipelines cho production.
- Graduate student nghiên cứu Speech AI.

## Stack kỹ thuật

- Static site generator: [mdBook](https://rust-lang.github.io/mdBook/), Rust-based, build nhanh.
- Math rendering: [mdbook-katex](https://github.com/lzanini/mdbook-katex) preprocessor (build-time, không phụ thuộc runtime MathJax).
- Typography: Source Serif 4 (body), Inter (heading), JetBrains Mono (code).
- Hosting: GitHub Pages, branch `gh-pages`.
- CI/CD: GitHub Actions.

## Cấu trúc thư mục

```
src/                  # mdBook markdown source
├── SUMMARY.md        # Navigation
├── index.md          # Trang chủ và hướng dẫn đọc
├── part0/            # Phần 0  - Nền tảng cổ điển (Chương 0)
├── part1/            # Phần I  - Cầu nối khái niệm và tín hiệu (Chương 1-3)
├── part2/            # Phần II - ASR (Chương 4-7)
├── part3/            # Phần III - TTS và codec (Chương 8-10)
├── part4/            # Phần IV - Speech LLM và Multimodal (Chương 11-13)
├── part5/            # Phần V  - Understanding và Translation (Chương 14-15)
├── part6/            # Phần VI - Tiếng Việt (Chương 16-17)
├── part7/            # Phần VII - Tools, Inference, Production (Chương 18-21)
├── appendices/       # Phụ lục A-F
└── references.md     # Tài liệu tham khảo tổng hợp

book.toml             # mdBook config
theme/                # custom.css, custom.js, katex-macros.txt
.github/workflows/    # publish-mdbook.yml
docs/                 # STYLE_GUIDE.md, research notes, progress trackers
```

## Build cục bộ

```bash
# Cài mdBook và preprocessor (cần Rust toolchain)
cargo install mdbook --version 0.4.40
cargo install mdbook-katex --version 0.9.4

# Build static site (output trong ./book/)
mdbook build

# Live preview
mdbook serve --open    # http://localhost:3000
```

## Văn phong và quy chuẩn biên tập

Sách tuân thủ tone *giảng viên Việt chuyên nghiệp*, chi tiết tại `docs/STYLE_GUIDE.md`. Mỗi chương khi rewrite cần đạt checklist 14 mục cuối tài liệu này, bao gồm: intuition trước formalism, bridge NLP↔Speech, evidence/SOTA discipline, KaTeX safety, em-dash policy.

## Theo dõi tiến độ

Trạng thái rewrite từng phần và từng chương được theo dõi trong `docs/REWRITE_PROGRESS.md`. Research notes hỗ trợ cho các chương frontier (Speech LLM, multimodal, production) lưu tại `docs/research-q4-2025-models.md`.

## Lịch sử kỹ thuật

Repo bắt nguồn từ `dung-vpt/speech-processing-book` (stack Quarto). Đã migrate qua ba stack: Quarto → MkDocs Material → mdBook (chốt). Snapshot Quarto được giữ lại ở branch `quarto-legacy`.

## License

MIT, xem [LICENSE](LICENSE).
