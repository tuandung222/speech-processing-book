# S-AI Style Guide

Tài liệu chuẩn biên tập cho cuốn sách **S-AI: Speech AI từ Signal Processing đến Full-Duplex Dialogue**.

**Đối tượng độc giả**: data scientist nền tảng **NLP/LLM** (BERT, GPT, Transformer) và **CV** chuyển sang **Speech AI**.

**Mục tiêu chất lượng**: chất lượng giảng viên đại học chuyên nghiệp, giàu năng lực sư phạm. Sách phải vừa chính xác về kỹ thuật, vừa truyền tải được trực giác và bối cảnh thực tiễn cho người mới.

---

## 1. Voice và Tone (bắt buộc)

### 1.1 Tone chuẩn: giảng viên Việt chuyên nghiệp

Toàn bộ chương phải được viết bằng giọng **một giảng viên đại học người Việt có kinh nghiệm**: tự nhiên, rõ ràng, nghiêm túc, có khả năng dẫn dắt sinh viên qua khái niệm khó mà không hạ thấp trình độ người đọc.

**Đặc điểm tone đúng**:

- Câu văn dứt khoát, có cấu trúc, không lan man.
- Dùng `ta`, `chúng ta`, `người đọc`, hoặc `bạn` ở mức trang trọng vừa phải; tránh lạm dụng đại từ nhân xưng.
- Đặt câu hỏi dẫn dắt khi cần định hướng tư duy, nhưng phải mang tính học thuật, không phải retoric kiểu marketing.
- Diễn đạt trực tiếp khi bàn về trade-off và hạn chế; không tô vẽ.
- Ưu tiên chính xác về kỹ thuật hơn là dễ thương về văn phong.

**Đặc điểm tone phải tránh**:

- Văn phong cợt nhã, đùa giỡn, hoặc kể chuyện cá nhân không phục vụ nội dung.
- Từ cảm thán biểu lộ phấn khích quá mức (`Bùm`, `cực kỳ elegant`, `đẹp ghê`).
- Câu mở bài kiểu trò chuyện suồng sã (`OK giờ ta`, `Anyway`, `Alright`, `Nói thật là`).
- Câu trấn an kiểu cá nhân hoá (`đừng hoảng nhé`, `đừng lo`, `bình tĩnh nào`).
- Bullet-dump khô khan, không có dòng nối hay diễn giải.
- Khẳng định tuyệt đối khi không có nguồn (`luôn luôn`, `chắc chắn`, `không bao giờ`) trừ khi đúng về mặt toán học.

### 1.2 Xây dựng trực giác trước, hình thức hoá sau

Mỗi khái niệm mới nên đi qua ba lớp:

1. **Trực giác**: vì sao bài toán này tồn tại, bài toán cố giải quyết điều gì, mâu thuẫn nào dẫn đến giải pháp này.
2. **Hình thức hoá**: phát biểu toán học rõ ràng, biến số được khai báo, ràng buộc rõ.
3. **Hệ quả thực tiễn**: ý nghĩa cho training, inference, deployment hoặc debugging.

**Đối lập (cần tránh)**:

> CTC loss được định nghĩa: $\mathcal{L}_{CTC} = -\log \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^{T} P(\pi_t | X)$. Cần lưu ý CTC giả định độc lập có điều kiện theo thời gian.

**Cách viết đạt chuẩn**:

> Trong bài toán nhận dạng giọng nói, ta thường có chuỗi audio frames dài hàng nghìn bước nhưng nhãn text chỉ vài chục ký tự. Khác với dịch máy seq2seq, ta không có alignment sẵn giữa frame và ký tự. CTC giải bài toán này bằng cách mở rộng vocabulary với một token đặc biệt `blank` ($\epsilon$), cho phép model phát ra một ký tự hoặc `blank` tại mỗi frame, sau đó **collapse** chuỗi này về nhãn cuối bằng cách lược bỏ ký tự lặp liền kề và bỏ `blank`.
>
> Loss CTC marginalize trên tập tất cả paths $\pi$ collapse về nhãn $Y$:
>
> $$\mathcal{L}_{CTC}(X, Y) = -\log \sum_{\pi \in \mathcal{B}^{-1}(Y)} \prod_{t=1}^{T} P(\pi_t \mid X)$$
>
> trong đó $X$ là chuỗi feature đầu vào, $T$ là số frames, $\mathcal{B}^{-1}(Y)$ là tập paths collapse về $Y$.
>
> Một giả định quan trọng cần ghi nhận: CTC giả định các output theo time step là độc lập có điều kiện cho input. Đây là điểm yếu mà RNN-Transducer sẽ khắc phục bằng predictor network, được trình bày ở mục sau.

### 1.3 Bắc cầu NLP↔Speech mỗi khái niệm mới

Vì độc giả mục tiêu đến từ NLP/LLM, mỗi khái niệm Speech lạ đều nên có một đoạn liên hệ tương ứng với khái niệm NLP đã quen thuộc.

Ví dụ:

- **Mel spectrogram** có thể được nhìn như một dạng biểu diễn dense, deterministic của audio, đóng vai trò tương đương feature handcrafted cho ASR, khác với learned embeddings của ngôn ngữ.
- **CTC blank token** đóng vai trò tương tự token đặc biệt trong NLP (ví dụ `<pad>` khi xử lý chuỗi có độ dài thay đổi), nhưng mang ý nghĩa "không phát ra ký tự nào tại frame này".
- **Codec tokens** (EnCodec, Mimi) là analog gần nhất của BPE tokens cho audio: biến tín hiệu liên tục thành chuỗi token rời rạc có vocabulary cỡ 1k-8k, cho phép áp dụng paradigm autoregressive LM.
- **Wav2Vec 2.0 / HuBERT self-supervised pretraining** lặp lại đúng triết lý của BERT MLM: masked prediction trên input bị che, học representation tổng quát từ dữ liệu không nhãn.

### 1.4 Tiên đoán và xử lý điểm khó hiểu

Khi gặp khái niệm phản trực giác hoặc dễ nhầm lẫn, dừng lại để giải thích trước khi đi tiếp. Cụm thường dùng:

> **Lưu ý dễ nhầm**
> Khái niệm $A$ và $B$ thoạt nhìn giống nhau nhưng khác biệt ở $\ldots$. Cụ thể, $A$ là $\ldots$, còn $B$ là $\ldots$.

Hoặc:

> **Vì sao điều này đúng?**
> Người đọc có thể đặt câu hỏi: tại sao không dùng phương án $X$? Lý do là $\ldots$.

Khi đoạn nội dung nặng về toán, mở đầu bằng một câu cảnh báo ngắn về độ khó và mục tiêu của đoạn:

> Phần này phát triển công thức đầy đủ cho forward-backward algorithm trong CTC. Mục tiêu là thu được công thức cho gradient của loss theo logit, từ đó hiểu vì sao CTC ổn định khi train sequence dài.

### 1.5 Tự nhiên trong mix Vietnamese-English

Cuốn sách chấp nhận và khuyến khích mix Vietnamese-English ở những chỗ thuật ngữ English là chuẩn ngành. Đây không phải vay mượn tuỳ tiện mà là cách diễn đạt chính xác hơn cho practitioner.

**Dùng English**:

- Thuật ngữ kỹ thuật chuẩn: `encoder`, `decoder`, `transformer`, `attention`, `embedding`, `tokenizer`, `pipeline`, `inference`, `latency`, `throughput`, `fine-tune`, `pretrain`, `deploy`, `streaming`, `endpointing`.
- Acronym: `ASR`, `TTS`, `NLP`, `LLM`, `MoE`, `KV cache`, `STFT`, `MFCC`, `CTC`, `RNN-T`, `S2TT`, `KWS`.
- Modern jargon: `in-context learning`, `prompt engineering`, `function calling`, `tool calling`, `hallucination`, `code-switching`, `retrieval-augmented`, `voice cloning`.
- Tên model, framework, công ty: `Whisper`, `Conformer`, `Wav2Vec 2.0`, `Moshi`, `Qwen3-Omni`, `WeNet`, `ESPnet`, `Triton`, `vLLM`, `Cartesia`, `Deepgram`.

**Dùng Vietnamese**:

- Khái niệm có translation phổ biến và tự nhiên: `âm thanh`, `tần số`, `giọng nói`, `ngữ nghĩa`, `tín hiệu`, `phổ`, `mô hình`, `huấn luyện`, `dự đoán`.
- Đoạn diễn giải, lập luận, dẫn dắt.
- Tên các phần, chương, mục.

**Cần tránh**:

- Dịch ép thuật ngữ chuẩn (ví dụ `siêu tham số` thay cho `hyperparameter`, `bộ giải mã` thay cho `decoder` khi đoạn văn đang nói về kiến trúc transformer cụ thể).
- Trộn lẫn không nhất quán (đôi chỗ `encoder` đôi chỗ `bộ mã hoá` trong cùng một mục).
- Câu lai kiểu cẩu thả (`I think rằng`, `Anyway thì`, `So là`).

Nguyên tắc cuối: nếu một thuật ngữ chuẩn ngành đã đi vào ngôn ngữ kỹ thuật Việt (`mô hình`, `huấn luyện`, `chương trình`), dùng tiếng Việt; nếu thuật ngữ vẫn là English trong communication chuyên môn hằng ngày (`encoder`, `latency`, `streaming`), giữ English.

### 1.6 Nhịp văn

Đa dạng nhịp giữa các đoạn để tránh đơn điệu:

- Đoạn nặng toán: viết chậm, từng dòng, định nghĩa biến số trước khi dùng.
- Đoạn trực giác: súc tích, có ví dụ cụ thể, có analogy ngắn.
- Đoạn so sánh, tổng kết: dùng bảng nếu phù hợp; nếu không, dùng paragraph có cấu trúc.
- Đoạn chuyển: một hai câu dẫn dắt sang chủ đề tiếp theo, không cần dài.

Tránh:

- Mọi đoạn cùng độ dài.
- Mọi câu cùng cấu trúc chủ ngữ + động từ + bổ ngữ.
- Đoạn dài quá 6-7 câu mà không có break (bảng, công thức, sub-heading).

### 1.7 Văn xuôi chảy thay vì bullet-dump

Bullet-list phù hợp cho catalog, so sánh tham số, danh sách tham chiếu. Bullet không phù hợp cho giải thích chính.

**Đối lập (cần tránh)**:

> CTC có 3 đặc điểm:
> - Thêm blank token
> - Output mỗi frame
> - Marginalize alignments

**Cách viết đạt chuẩn**:

> CTC khác seq2seq cổ điển ở ba thiết kế cốt lõi. Trước hết, vocabulary được mở rộng thêm một **blank token** $\epsilon$ với ý nghĩa "không phát ra ký tự nào tại frame này". Thứ hai, model output một phân phối xác suất tại mỗi frame, không có khái niệm "decoder step" riêng biệt như trong seq2seq. Thứ ba, hàm loss marginalize trên tất cả paths khả dĩ collapse về nhãn, thay vì cố định một alignment.

### 1.8 Bảng cho tham chiếu, văn xuôi cho lập luận

Dùng bảng khi:

- So sánh tham số (kích thước model, accuracy, latency).
- Catalog tham chiếu (sample rate, model checkpoint, API).
- Bảng tra chéo (NLP↔Speech mapping).

Không dùng bảng khi:

- Trình bày lập luận chính (dùng paragraph).
- Diễn giải từng bước (dùng paragraph có thứ tự).
- Định nghĩa khái niệm trung tâm.

### 1.9 Câu hỏi dẫn dắt mang tính học thuật

Câu hỏi mở đầu một mục có thể được dùng, nhưng phải thật sự dẫn vào nội dung, không trang trí.

**Tốt**:

> Trước khi đi vào kiến trúc Conformer, ta cần trả lời một câu hỏi: vì sao Transformer thuần khi áp dụng cho audio thường kém hơn so với khi áp dụng cho text?

**Tránh**:

> Bạn có thấy điều gì đặc biệt ở đây không?
> Đoán xem chuyện gì sẽ xảy ra tiếp theo?

---

## 2. Bảng từ vựng NLP↔Speech (bắt buộc tham chiếu)

Khi gặp các thuật ngữ trong bảng này, luôn nhắc lại liên hệ với khái niệm NLP tương ứng ít nhất một lần trong chương.

| Speech AI | NLP analogue | Bridge explanation |
|---|---|---|
| Mel spectrogram | Static word embedding | Biểu diễn handcrafted, deterministic, không tự học. |
| Wav2Vec 2.0, HuBERT | BERT MLM | Self-supervised pretrain với masked prediction. |
| EnCodec, Mimi codec tokens | BPE tokens | Vocabulary rời rạc cỡ 1k-8k, học từ data bằng VQ-VAE. |
| CTC loss | seq2seq cross-entropy | Loss cho sequence input-output không có alignment sẵn. |
| CTC blank token | `<pad>` / `<mask>` | Token đặc biệt, ở CTC mang ý nghĩa "no emission". |
| RNN-Transducer | Transformer decoder | Streaming-friendly seq2seq với predictor + joint network. |
| Conformer | Transformer encoder | Hybrid CNN + self-attention, optimized cho audio sequence dài. |
| Whisper | T5 / BART | Encoder-decoder, multitask, pretrained ở quy mô lớn. |
| VITS, F5-TTS | GPT / diffusion generator | End-to-end generative cho audio. |
| Phoneme | Subword (BPE chunk) | Đơn vị ngôn ngữ nhỏ hơn từ. |
| Forced alignment | Token-to-char alignment | Map frame audio sang đơn vị text. |
| Voice Activity Detection | Sentence boundary detection | Phân đoạn stream liên tục. |
| Speaker embedding | User / persona embedding | Vector định danh cho conditional generation. |
| Streaming ASR | Causal LM | Dự đoán online không cần future context. |
| Full-duplex dialogue | Multi-turn chat overlap | Đối thoại realtime hai chiều đồng thời. |
| Audio LLM (Moshi, Qwen2-Audio) | Multimodal LLM | LLM với input/output audio. |

---

## 3. Quy ước toán học

### 3.1 Biến số

| Loại | Quy ước | Ví dụ |
|---|---|---|
| Scalar (time, frequency, dimension) | chữ thường nghiêng | $t$, $f$, $d$ |
| Vector | chữ thường đậm | $\mathbf{x}$, $\mathbf{h}$ |
| Matrix | chữ hoa đậm | $\mathbf{W}$, $\mathbf{H}$ |
| Tập hợp | mathcal | $\mathcal{V}$, $\mathcal{L}$ |
| Biến ngẫu nhiên | chữ hoa nghiêng | $X$, $Y$ |
| Xác suất | $P(\cdot)$ hoặc $p(\cdot)$ | $P(y \mid x)$ |

### 3.2 KaTeX safety

Cuốn sách dùng `mdbook-katex` preprocessor. Quy tắc bắt buộc để tránh lỗi parse:

1. **Underscore trong `\text{}`**: bắt buộc escape, `\text{frame\_length}` không phải `\text{frame_length}`.
2. **Dollar sign trong prose**: tránh viết `$0`, `$5/min`, `$15-30`; dùng `0 USD`, `5 USD/min`, `15-30 USD` để không kích hoạt math span ngoài ý muốn.
3. **Display math indentation**: nếu `$$ ... $$` nằm trong admonition hoặc list, dấu `$$` đóng phải ở cùng cấp indentation với dấu mở.
4. **Macro**: định nghĩa macro chung trong `theme/katex-macros.txt`, không lặp lại trong chương.
5. **Inline math không xuống dòng**: `$\ldots$` không được chứa newline.

### 3.3 Đơn vị đo

- Thời gian: `s`, `ms` (không dùng `mili-giây` trong công thức, dùng trong prose).
- Tần số: `Hz`, `kHz`.
- Kích thước model: `M params` (triệu), `B params` (tỉ).
- Latency: `ms` cho `< 1s`, `s` cho lớn hơn.
- Bandwidth: `kbps`, `Mbps`.

---

## 4. Cấu trúc chương chuẩn

Mỗi chương nên có:

1. **Mở đầu**: nêu vì sao chương này quan trọng cho độc giả NLP/LLM/CV.
2. **Bản đồ chương**: section breakdown ngắn (`> Cấu trúc chương`).
3. **Các phần nội dung** đánh số `## Phần 1`, `## Phần 2`, ...
4. **Tóm tắt**: các ý chính, không quảng cáo.
5. **Chương tiếp theo**: một đoạn cầu nối.
6. **Tài liệu tham khảo**: danh sách cuối chương, kèm URL khi có.

Section trong chương dùng heading `##` cho phần, `###` cho mục, `####` cho mục con. Tránh đào sâu quá `####`.

---

## 5. Evidence và SOTA discipline

Trong lĩnh vực đang chuyển động nhanh như Speech AI, kỷ luật về evidence là yêu cầu bắt buộc.

**Khi nêu số liệu** (WER, BLEU, MOS, latency, pricing, model release date):

- Phải có nguồn rõ: paper, blog post, release notes, công ty công bố.
- Nếu là estimate, đánh dấu rõ: `ước lượng dựa trên`, `estimated from public pricing`.
- Time-stamp khi nội dung có thể thay đổi nhanh: pricing, model release.

**Khi nêu SOTA**:

- Tránh khẳng định tuyệt đối (`SOTA hiện tại`) trừ khi có benchmark cụ thể và thời điểm.
- Đặt context: SOTA trên benchmark nào, với điều kiện nào.
- Cross-check ≥2 nguồn cho claim nhạy cảm.

**Cần tránh**:

- `Đây là model tốt nhất từng có.`
- `Whisper giải quyết hoàn toàn bài toán ASR.`
- `Voice cloning đã đạt mức không phân biệt được với người thật.`

**Diễn đạt đúng**:

- `Trên VLSP 2020 Task-1, PhoWhisper-large đạt WER 10.8%, giảm khoảng 35% so với Whisper-large-v3 baseline (Le et al., ICLR 2024 Tiny Papers).`
- `Theo public pricing của ElevenLabs tháng 11/2025, gói Voice Agent có chi phí khoảng 0.08-0.12 USD/min.`

---

## 6. Em-dash và dấu câu

Em-dash (`—`, `--`) có thể gây lỗi parse trong một số trình renderer Markdown và làm văn phong nặng nề khi dùng quá nhiều. Quy tắc:

- **Heading, mục**: được phép dùng em-dash để phân tách tên Phần với mô tả ngắn, ví dụ `Phần I — Nền tảng tín hiệu`.
- **Prose**: ưu tiên dùng dấu phẩy, dấu hai chấm, hoặc ngoặc đơn thay cho em-dash.
- **Tránh** chuỗi em-dash trong cùng một câu.

---

## 7. Hình ảnh và sơ đồ

- Đặt trong thẻ `<figure>` có `id`, kèm `<figcaption>`.
- File ảnh ở cùng thư mục với chương hoặc trong `src/<part>/assets/`.
- Ưu tiên SVG cho sơ đồ, PNG cho screenshot, JPEG cho ảnh thật.
- Caption mô tả nội dung, không quá súc tích đến mức không hiểu khi đọc rời.

---

## 8. Code listing

- Python 3.10+, type hint đầy đủ.
- PyTorch 2.x + `torchaudio`.
- Inline tensor shape comments: `# [batch, seq, dim] - dtype`.
- Imports ở đầu snippet hoặc nhắc lại nếu snippet nằm xa snippet trước.
- Tránh code dài quá ~50 dòng; nếu cần dài, chuyển sang Appendix D và link tới.

---

## 9. Tham chiếu

- Bibliographic style ưu tiên `Author, Year, Title, Venue`.
- URL bọc trong `<...>` cho mdBook render đúng.
- Đặt section `## Tài liệu tham khảo` cuối mỗi chương.
- Khi reference nhiều lần trong chương, có thể dùng footnote `[^key]` với định nghĩa cuối file.

---

## 10. Checklist chất lượng cho mỗi chương

Trước khi commit một chương mới hoặc rewrite chương cũ, đối chiếu:

- [ ] Mở đầu nêu rõ vì sao chương quan trọng cho độc giả NLP/LLM/CV.
- [ ] Trực giác đi trước hình thức hoá.
- [ ] Mọi biến số trong công thức đều được khai báo.
- [ ] Có ít nhất một analogy NLP↔Speech ở khái niệm trung tâm.
- [ ] Section thực hành: training, inference, deployment, hoặc debugging.
- [ ] Section limitations và failure modes.
- [ ] Vietnamese context khi liên quan.
- [ ] Tone giảng viên chuyên nghiệp, không cợt nhã.
- [ ] Không có phrase casual cấm: `Bùm`, `đừng hoảng`, `OK giờ ta`, `Anyway`, `Alright`, `Nói thật là`, `cực kỳ elegant`.
- [ ] Không có SOTA claim không có nguồn.
- [ ] Không có em-dash trong prose (chỉ heading nếu cần).
- [ ] Không có KaTeX error sau `mdbook build`.
- [ ] Tài liệu tham khảo cuối chương đầy đủ.
- [ ] Liên kết, hình ảnh, bảng đều render đúng.
