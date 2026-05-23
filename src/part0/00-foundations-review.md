# Chương 0: Nền tảng cổ điển — Ngôn ngữ học, Âm học, DSP và ML truyền thống

> **Trạng thái biên soạn**
>
> Chương này đang được biên soạn trong giai đoạn rewrite Phần 0 và Phần I. Nội dung chính thức sẽ bao gồm bốn mảng nền tảng cần thiết cho người đọc đến từ NLP/LLM hoặc CV, trước khi bước vào các chương deep learning hiện đại.

## Mục tiêu của chương

Chương 0 đóng vai trò *review chapter* dành cho người đọc chưa có background về xử lý tín hiệu, ngôn ngữ học, hoặc machine learning truyền thống. Mục tiêu là cung cấp đủ vốn từ vựng và mô hình tư duy cơ bản để tiếp cận các chương sau mà không bị bí ở những khái niệm "ai cũng giả định bạn biết".

Nội dung sẽ phủ bốn mảng:

1. **Ngôn ngữ học cơ bản**: phonetics, phonology, phoneme, allophone, prosody, cùng đặc thù của tiếng Việt (sáu thanh điệu, ba phương ngữ).
2. **Âm học và thính giác**: sóng âm, hệ thính giác con người, source-filter model, mel scale và bối cảnh ra đời.
3. **Digital signal processing nền tảng**: định lý lấy mẫu Nyquist-Shannon, lượng tử hoá, convolution, filtering, trực giác về biến đổi Fourier và STFT (chi tiết kỹ thuật được phát triển trong Chương 2).
4. **Machine learning cổ điển cho speech**: DTW, HMM, GMM, hybrid GMM-HMM ASR, i-vectors, cùng lý do deep learning thay thế các cách tiếp cận này.

Mỗi mảng sẽ có một đoạn liên hệ với khái niệm tương đương trong NLP/LLM để người đọc dễ định vị.

## Khi nào nên đọc chương này

- Đọc nếu bạn đến từ NLP/LLM thuần và chưa từng học một khoá xử lý tín hiệu hoặc ngôn ngữ học.
- Đọc nếu bạn cần ôn lại kiến thức HMM/GMM trước khi vào Chương 4 (ASR Foundations) và Chương 5 (Modern ASR).
- Có thể bỏ qua nếu bạn đã quen với mel filterbank, MFCC, Fourier, và đã từng làm việc với các framework ASR truyền thống như Kaldi.

## Tham khảo gợi ý

- Jurafsky, D. và Martin, J. H. (2024). *Speech and Language Processing* (SLP3). <https://web.stanford.edu/~jurafsky/slp3/>.
- Aalto University. *Speech Processing Book*. <https://speechprocessingbook.aalto.fi/>.
- Rabiner, L. R. và Schafer, R. W. (2010). *Theory and Applications of Digital Speech Processing*. Pearson.

Phiên bản chi tiết của chương đang được hoàn thiện và sẽ thay thế phần mô tả ngắn này trong các commit kế tiếp.
