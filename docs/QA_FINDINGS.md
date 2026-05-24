# Báo cáo Đánh giá Chất lượng & Visual (Phase QA-0)

Thực hiện kiểm tra trực tiếp toàn bộ tài liệu S-AI để xác định các vấn đề về hiển thị công thức toán học, trình bày code, lựa chọn font chữ, typography và sơ đồ minh hoạ.

---

## 1. Vấn đề Hiển thị Công thức Toán học (KaTeX)

Dù trình biên dịch `mdbook-katex` không báo lỗi lúc build (0 parse errors), khi render trên trình duyệt thực tế, có một số vấn đề hiển thị ảnh hưởng đến tính khoa học và thẩm mỹ:

### 1.1 Ký tự gạch dưới bị hiển thị backslash vô lý
Trong Chương 2 và Chương 16, chúng ta có các ký tự biến số như `\text{frame\_length}` hoặc `\text{reference\_audio}`. KaTeX xử lý các cụm này không nhất quán:
- Một số trình duyệt hiển thị rõ cả ký tự backslash thoát: `frame\_length` (không phù hợp với chuẩn trình bày học thuật).
- **Giải pháp**: Đổi toàn bộ `\text{frame\_length}` thành `\mathrm{frame\_length}` hoặc `\text{frame length}` (không dùng gạch dưới bên trong block text toán học).

### 1.2 Căn lề và khoảng cách hiển thị công thức
- Các khối công thức toán dạng display `$$ ... $$` hiển thị quá sát với phần văn bản (margin-top/bottom nhỏ).
- **Giải pháp**: Tăng margin của `.katex-display` lên `1.2em 0` và bổ sung đường viền bên trái cực nhẹ để nhấn mạnh khối công thức trong tài liệu dạng nghiên cứu.

---

## 2. Vấn đề Trình bày Code Blocks

Hệ thống highlight mặc định của mdBook khá đơn điệu và thiếu độ tương phản:

### 2.1 Thiếu Line Numbers và Copy Button chuyên nghiệp
- Trình duyệt hiển thị code block không có đánh số dòng rõ ràng khiến việc đối chiếu giải thích rất khó khăn.
- Kiểu font code mặc định hơi nhỏ và khoảng cách dòng (`line-height`) bị khít, gây cảm giác khó đọc khi có các comment ghi chú shape của tensor.

### 2.2 Giải pháp Overhaul Code Rendering
- Cập nhật CSS để biến đổi giao diện code block: sử dụng tone nền ấm hơn (warm grey) và bo góc mềm mại (`border-radius: 6px`).
- Tăng cỡ font code lên `0.85rem` và tăng chiều cao dòng lên `1.5` để các comments tensor shape `# [B, T, D]` đứng thẳng hàng, dễ nhìn.
- Thêm hiệu ứng hover sáng nhẹ cho block code đang được đọc.

---

## 3. Typography & Sắp xếp Font Chữ (Distill.pub Style)

Typography ban đầu chưa đạt chuẩn tài liệu nghiên cứu cao cấp do các nguyên nhân:

### 3.1 Google Fonts Import gây chậm và FOUC
- Việc sử dụng `@import` trực tiếp Google Fonts làm chậm tốc độ tải trang trên GitHub Pages. Khi trang vừa load, font hệ thống hiển thị trước rồi bị giật (Flash of Unstyled Text) khi font Source Serif 4 tải xong.
- **Giải pháp**: Thay thế bằng cách load font thông qua thẻ `<link>` tối ưu trong HTML head hoặc tự host các font core.

### 3.2 Khuyết điểm trong Layout & Cấu trúc Khối
- **Kích thước Content**: Có sự xung đột giữa biến tối đa `--content-max-width: 920px` và `.content max-width: 820px`. Cần chuẩn hoá về `840px` cho văn bản thông thường để mắt không phải liếc quá rộng khi đọc, nhưng cho phép các bảng so sánh lớn và sơ đồ mở rộng ra tối đa `1040px` (đúng chuẩn Distill.pub).
- **Gạch chân H2**: Việc gạch chân dưới các tiêu đề H2 bằng nét liền mảnh `border-bottom: 1px solid` trông rất giống giáo trình cũ những năm 2010, làm giảm sự hiện đại của website.
- **Giải pháp**: Loại bỏ đường kẻ này. Sử dụng tỷ lệ vàng về kích thước, độ đậm (font-weight) và khoảng cách (letter-spacing) để phân cấp thông tin một cách tự nhiên và sang trọng.

---

## 4. Visualization & Sơ đồ Minh hoạ

Các hình ảnh minh hoạ hiện tại sử dụng thẻ `<img>` trỏ đến các file PNG tĩnh tự tạo chất lượng thấp, không đồng bộ với giao diện và khó chỉnh sửa:

### 4.1 Điểm yếu của ảnh tĩnh PNG
- Khi chuyển đổi giao diện Dark Mode / Light Mode, các ảnh PNG có nền trắng sẽ bị lọt thỏm, gây loá mắt cho người đọc ở chế độ ban đêm.
- Các sơ đồ ASCII art tự vẽ bằng ký tự `─ ↓ │` bị lệch dòng trên một số màn hình di động.

### 4.2 Giải pháp Visual mới
- **Tích hợp Mermaid.js**: Cấu hình preprocessor `mdbook-mermaid` để vẽ trực tiếp các sơ đồ quan hệ bằng mã code. Mermaid sẽ tự động render ra SVG sắc nét ở mọi độ phân giải và tự đổi màu theo theme Light/Dark cực kỳ mượt mà.
- Thiết kế các sơ đồ kiến trúc trọng tâm:
  - Sơ đồ Tiến hoá của Speech LLM (Chương 11).
  - Sơ đồ Luồng Latency Budget cho Real-Time Agent (Chương 20).
  - Sơ đồ Phân cấp Audio Codecs (Chương 10).

---

## 5. Review Cấu trúc Sách & Mục lục

Rà soát lại toàn bộ 22 chương hiện có, chúng tôi phát hiện một số điểm chưa hợp lý về mặt logic:

1. **Vị trí Chương 21 (Wake-Word Detection)**: Hiện tại chương này nằm ở Phần VII (Tools & Production). Tuy nhiên, về mặt bản chất khoa học, Wake-Word (Keyword Spotting) thuộc nhóm bài toán **Speech Classification / Analysis** (Phần V). Việc đặt nó cạnh "Inference Engines" và "Production Systems" làm loãng tính hệ thống.
   - **Đề xuất**: Di chuyển Chương 21 vào Phần V, đứng ngay sau Chương 14 (Speech Classification).
2. **Trùng lặp khái niệm Codec Tokens**: Codec tokens được giới thiệu sơ bộ ở Chương 3 (Speech Representations) và viết rất chi tiết ở Chương 10 (Audio Codecs).
   - **Đề xuất**: Định rõ giới hạn: Chương 3 chỉ giới thiệu ở mức độ "khái niệm tương đương với BPE token trong NLP", còn Chương 10 mới đi sâu vào toán học Residual Vector Quantization (RVQ) và cấu trúc nén của EnCodec/Mimi.
3. **Thiếu mảng Speech Enhancement**: Bộ tài liệu hiện có ASR, TTS, Translation, nhưng thiếu mảng Speech Enhancement (Denoising, Dereverberation, Speaker Separation) vốn là tiền xử lý bắt buộc trong mọi hệ thống voice thực tế.
   - **Đề xuất**: Bổ sung một mục chi tiết hoặc một chương phụ về Speech Enhancement trong Phần VII để hoàn thiện kỹ năng thực chiến.

---

## 6. Trạng thái sau vòng QA A-Z (2026-05-25)

Vòng review, rewrite và verify hàng loạt đã xử lý các vấn đề chính được ghi nhận ở Phase QA-0:

- **Math rendering**: quy tắc KaTeX đã được đồng bộ trong `docs/STYLE_GUIDE.md`; không dùng underscore trực tiếp bên trong `\text{}` khi nhãn có thể viết dạng văn xuôi.
- **Build/deploy**: GitHub Actions đã cài thêm `mdbook-mermaid` để khớp với `book.toml`.
- **Content rigor**: các claim kiểu `SOTA`, `best`, `outperform` đã được hạ giọng thành `frontier`, `baseline mạnh`, hoặc `theo technical report`, kèm nhắc lại điều kiện benchmark khi cần.
- **Vietnamese chapters**: Chương 16 và 17 đã được chỉnh lại theo hướng thận trọng hơn về benchmark, model thương mại và dữ liệu công khai.
- **Production chapter**: Chương 20 đã được rewrite phần mở đầu, disclaimer, latency/concurrency/cost/SLA để có giọng giảng viên chuyên nghiệp hơn.
- **Visualization**: các hình kiến trúc trọng tâm và ASCII pipeline đã được thay bằng Mermaid diagrams; scanner public không còn `<img src="fig-...">`.
- **Final scan public content**: `TBD = 0`, forbidden casual phrases = 0, static figure placeholders = 0.
