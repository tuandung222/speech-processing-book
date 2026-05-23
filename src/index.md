# Lời nói đầu

## Tại sao cuốn sách này?

Trong thập kỷ qua, cộng đồng NLP đã chứng kiến sự bùng nổ của Large Language Models (LLMs) - từ BERT [^devlin2019bert] đến GPT-4 [^openai2024gpt4o], từ text generation đến multimodal reasoning. Nhưng có một lĩnh vực mà ranh giới giữa "ngôn ngữ" và "tín hiệu vật lý" trở nên mờ nhạt: **Speech AI**.

Speech AI không chỉ là "thêm một modality" vào LLM. Nó đòi hỏi sự hiểu biết sâu về:

- **Signal processing**: Biến đổi Fourier, mel spectrogram, MFCC [^davis1980comparison] [^oppenheim1999discrete]
- **Sequence alignment**: CTC loss [^graves2006connectionist], attention-based alignment, transducers [^graves2012sequence]
- **Generative modeling**: VAE [^kingma2014vae], normalizing flows [^rezende2015normalizing], flow matching [^lipman2023flow], GAN [^goodfellow2014generative]
- **Neural audio compression**: Vector quantization [^oord2017vqvae], residual VQ, codec tokens [^defossez2022encodec]
- **Self-supervised learning**: Wav2Vec 2.0 [^baevski2020wav2vec], HuBERT [^hsu2021hubert], WavLM [^chen2022wavlm]
- **Real-time systems**: Streaming inference, latency optimization, voice pipelines

Cuốn sách này được viết cho **CV/NLP data scientists** muốn trở thành **speech data scientists** chuyên nghiệp. Đây là tài liệu nội bộ toàn diện, bao quát từ nền tảng lý thuyết đến triển khai production.

## Đối tượng Độc giả

Cuốn sách được thiết kế cho:

1. **CV/NLP Data Scientists** muốn chuyển sang speech modality
2. **LLM Researchers** quan tâm đến multimodal models (Qwen2.5-Omni, Moshi, GPT-4o)
3. **ML/AI Engineers** xây dựng voice AI pipelines cho production
4. **Graduate students** nghiên cứu speech processing

**Prerequisites:**

- Transformer architecture [^vaswani2017attention] (self-attention, encoder-decoder)
- LLM training basics (tokenization, pre-training, fine-tuning)
- Python & PyTorch proficiency
- Linear algebra & probability fundamentals

## Cấu trúc Cuốn sách

Cuốn sách được thiết kế để đọc theo lộ trình phù hợp với background của bạn, không nhất thiết phải đọc tuyến tính từ đầu đến cuối. Mỗi phần phục vụ một mục đích sư phạm riêng. Xem mục **Hướng dẫn Đọc** ngay sau đây để chọn lộ trình.

Cuốn sách gồm **8 phần** (bao gồm Phần 0 nền tảng cổ điển) và **6 phụ lục**, tổng cộng **22 chương** (Chương 0 đến Chương 21), đang được mở rộng dần.

### Phần 0 — Nền tảng cổ điển (mới)

Dành cho audience NLP/LLM thiếu kiến thức ngôn ngữ học, âm học, hoặc DSP. Đây là chương review nhằm bù lấp gap kiến thức cơ bản trước khi vào deep learning hiện đại. Bạn có thể skip nếu đã quen với phonetics, Fourier, HMM, GMM.

- **Chương 0**: Linguistics, Acoustics, DSP, và Traditional ML (HMM-GMM, DTW, i-vectors)

### Phần I — Cầu nối khái niệm và nền tảng tín hiệu (Chương 1-3)

Phần quan trọng nhất để định hướng tư duy. Phần I xây dựng cây cầu giữa NLP/LLM bạn đã biết và Speech AI bạn sắp học. Sau Phần I, mọi thuật ngữ chuyên ngành sẽ không còn xa lạ.

- **Chương 1**: Từ NLP đến Speech — Cầu nối khái niệm
- **Chương 2**: Audio Signal Fundamentals (DFT, STFT, Mel filterbank, MFCC, SpecAugment)
- **Chương 3**: Speech Representations (SSL backbones, contrastive learning, codec tokens)

### Phần II — Nhận dạng giọng nói (ASR) (Chương 4-7)

Bài toán Speech-to-Text, bài toán Speech AI lâu đời nhất, có nhiều thành tựu nhất, và là cửa ngõ thực tiễn vào ngành.

- **Chương 4**: ASR Foundations — CTC, Sequence-to-Sequence with Attention, RNN-Transducer
- **Chương 5**: Modern ASR Architectures — Conformer, Zipformer, E-Branchformer
- **Chương 6**: Whisper & Canary — large-scale weakly-supervised ASR case studies
- **Chương 7**: Streaming ASR — kiến trúc và chiến lược cho real-time

### Phần III — Tổng hợp giọng nói (TTS) và codec (Chương 8-10)

Bài toán Text-to-Speech và các neural audio codec, là nền tảng cho voice cloning, voice agent, và Speech LLMs.

- **Chương 8**: TTS Foundations — Tacotron 2, FastSpeech 2, vocoders (HiFi-GAN, BigVGAN)
- **Chương 9**: End-to-End TTS — VITS, NaturalSpeech 3, F5-TTS, voice cloning với VALL-E
- **Chương 10**: Audio Codecs — EnCodec, DAC, Mimi, SpeechTokenizer, residual vector quantization

### Phần IV — Speech LLMs và Multimodal (Chương 11-13)

Phần "frontier" nhất, là điểm gặp gỡ giữa Speech AI và LLM revolution. Nếu bạn từ NLP/LLM, đây là phần bạn sẽ thấy quen thuộc và thú vị nhất.

- **Chương 11**: Speech Language Models — AudioLM, VALL-E, Moshi, Qwen2-Audio, Qwen3-Omni
- **Chương 12**: Multimodal Omni Models — Qwen3.5-Omni, Gemini 3, GPT-Realtime, native multimodal
- **Chương 13**: Full-Duplex Dialogue — Moshi architecture, conversational dynamics

### Phần V — Speech Understanding, Translation, và Wake-word (Chương 14-15, 21)

Các bài toán speech analysis (classification, translation, wake-word detection) là application layer của các kỹ thuật ở Phần I-IV.

- **Chương 14**: Speech Classification — Speaker ID, emotion recognition, language ID, audio events
- **Chương 15**: Speech Translation — S2TT, S2ST, simultaneous, SeamlessM4T deep dive
- **Chương 21**: Wake-Word Detection (Keyword Spotting) — "Hey Siri" và các giải pháp công nghiệp

### Phần VI — Tiếng Việt (Chương 16-17)

Phần dành riêng cho thị trường và đặc thù tiếng Việt. Bao gồm phân tích các công ty Việt Nam (VinAI, ZaloAI, FPT.AI, VinFast, Trusting Social) và đánh giá khả năng của mô hình hiện tại.

- **Chương 16**: Vietnamese Speech Processing — tonal features, code-switching, industry landscape
- **Chương 17**: Vietnamese Datasets & Benchmarks — VLSP, VIVOS, CMV-Vi, PhoWhisper

### Phần VII — Tools, Inference, Production (Chương 18-20)

Phần dành cho engineer triển khai thực tế. Khác Phần I-IV (focus lý thuyết và mô hình), Phần VII focus vào **làm sao deploy được vào production reliably**.

- **Chương 18**: Training Frameworks — WeNet, ESPnet, NeMo, SpeechBrain, K2/Icefall
- **Chương 19**: Inference Engines + Tooling — TensorRT-LLM, vLLM, Triton, ONNX Runtime, audio preprocessing libraries, augmentation, WeNet deep guide
- **Chương 20**: Production Speech Systems — voice agent architectures, latency budget, cost engineering, observability, deployment patterns

### Phụ lục (Appendices A-F)

Reference material để tra cứu khi cần, không cần đọc tuần tự.

- **A**: Notation Reference — quy ước ký hiệu toán học
- **B**: Mathematical Proofs — chứng minh chi tiết cho CTC forward-backward, mel filterbank, etc.
- **C**: NLP-Speech Concept Mapping — bảng tra cứu khái niệm (mở rộng từ Bảng 3.1 của Chương 1)
- **D**: Code Listings — full implementations dài
- **E**: Vietnamese Speech Resources — collection links, datasets, công cụ
- **F**: Tool Comparison Matrices — bảng so sánh framework, model, library

### Dependencies giữa các phần (đọc theo thứ tự ưu tiên)

```
Phần 0 ←─── (optional foundation for non-DSP background)
   │
Phần I ───→ Phần II ───→ Phần III ───→ Phần IV
   │           │              │            │
   │           ↓              ↓            ↓
   └────→ Phần V ─────────────────────────┘
                     │
                     ↓
              Phần VI (Vietnamese specifics)
                     │
                     ↓
              Phần VII (Production deployment)
```

Phần I là **prerequisite cho tất cả** (trừ Phần 0). Phần II-IV có thể đọc song song (mỗi phần độc lập về mặt mô hình). Phần V-VI-VII đều dựa trên Phần I-IV.

## Hướng dẫn Đọc

### Đến từ LLM/GPT research

1. **Chương 1** (NLP-to-Speech Bridge): bắt đầu từ đây.
2. **Chương 11-13** (Speech LLMs, Multimodal, Full-Duplex): lãnh thổ quen thuộc nhất.
3. **Chương 2-3** (Audio Fundamentals, Representations): bổ sung signal processing.
4. Các chương còn lại theo thứ tự.

### Đến từ NLP/BERT research

1. **Chương 1**, **Chương 2-3**, **Chương 4** (ASR ~ seq2seq translation).
2. **Chương 5-7** (Modern ASR), **Chương 8-10** (TTS & Codecs), **Chương 11-13** (Speech LLMs).

### Đến từ CV/Computer Vision

1. **Chương 2** (signal processing tương tự image features).
2. **Chương 14** (speech classification tương tự image classification).
3. **Chương 3** (self-supervised learning, tương đương SimCLR/DINO cho audio).

### Cần triển khai production ngay

1. **Chương 18-20** (Training frameworks, Inference engines, Production systems).
2. **Phụ lục F** (Tool comparison matrices).
3. Quay lại lý thuyết khi cần.

### Đọc toàn bộ (comprehensive)

Chương 0 đến Chương 21 theo thứ tự. Đây là lộ trình được khuyến nghị cho người đọc lần đầu tiếp cận Speech AI một cách hệ thống.

### Thiếu nền tảng signal processing hoặc HMM-GMM

Đọc **Chương 0** trước. Chương 0 cung cấp linguistics, acoustics, DSP và ML cổ điển ở mức ôn tập, đủ để hiểu các chương sau mà không cần đi quá sâu vào textbook truyền thống.

## Conventions

Xuyên suốt cuốn sách, chúng tôi sử dụng các conventions sau:

> **📝 Intuition**
>
> Giải thích trực giác về một khái niệm phức tạp.



> **⚠️ Hardware/Latency Warning**
>
> Cảnh báo về performance, latency, memory bottleneck, hoặc hardware constraints trong production.



> **💡 NLP Parallel**
>
> Mapping giữa khái niệm NLP quen thuộc và speech equivalent.



**Code style:**

```python
# Mọi tensor operation đều có inline shape & dtype comments
x: Tensor  # [batch, time, features] - torch.float32
mel: Tensor  # [batch, n_mels, n_frames] - torch.float32
codes: Tensor  # [batch, n_codebooks, T] - torch.int64
```

**Math notation:**

- Tất cả display equations được đánh nhãn: `<a id="eq-label"></a>`
- Cross-references: `[Phương trình](#eq-label)`, `[Hình](#fig-label)`, `[Bảng](#tbl-label)`
- Xem **Phụ lục A** để biết đầy đủ notation conventions

## Nguồn Tham khảo

Cuốn sách được xây dựng dựa trên **190+ papers và tài liệu** quan trọng trong lĩnh vực Speech AI, bao gồm:

- **ASR**: Whisper [^radford2023robust], Conformer [^gulati2020conformer], RNN-T [^graves2012sequence], Canary [^nvidia2024canary]
- **TTS**: VITS [^kim2021conditional], VALL-E [^wang2023valle], F5-TTS [^chen2024f5tts], CosyVoice [^du2024cosyvoice]
- **Audio Codecs**: EnCodec [^defossez2022encodec], DAC [^kumar2024dac], Mimi [^defossez2024mimi]
- **Speech LLMs**: AudioLM [^borsos2023audiolm], Qwen2-Audio [^chu2023qwen2audio], Moshi [^defossez2024moshi]
- **Multimodal**: Qwen2.5-Omni [^xu2025qwen25omni], Gemini [^team2024gemini]
- **Self-Supervised**: Wav2Vec 2.0 [^baevski2020wav2vec], HuBERT [^hsu2021hubert], WavLM [^chen2022wavlm]
- **Speech Translation**: SeamlessM4T [^communication2023seamlessm4t]
- **Vietnamese**: PhoWhisper [^nguyen2024phowhisper], PhoBERT [^nguyen2020phobert]
- **Textbooks**: SLP3 [^jurafsky2024speech], Spoken Language Processing [^huang2001spoken]
- **Training Tools**: WeNet [^zhang2022wenet], ESPnet [^watanabe2018espnet], NeMo [^kuchaiev2019nemo], SpeechBrain [^ravanelli2021speechbrain]



---

<!-- References (auto-generated from .bib) -->
[^devlin2019bert]: Devlin, Jacob and Chang, Ming-Wei and Lee, Kenton and Toutanova, Kristina, "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", North American Chapter of the Association for Computational Linguistics
[^openai2024gpt4o]: {OpenAI}, "GPT-4o System Card", OpenAI Technical Report
[^davis1980comparison]: Davis, Steven and Mermelstein, Paul, "Comparison of Parametric Representations for Monosyllabic Word Recognition", IEEE Transactions on Acoustics, Speech, and Signal Processing
[^oppenheim1999discrete]: Oppenheim, Alan V and Schafer, Ronald W, "Discrete-Time Signal Processing"
[^graves2006connectionist]: Graves, Alex and Fern{\'a}ndez, Santiago and Gomez, Faustino and Schmidhuber, J{\"u}rgen, "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks", International Conference on Machine Learning
[^graves2012sequence]: Graves, Alex, "Sequence Transduction with Recurrent Neural Networks", arXiv preprint arXiv:1211.3711
[^kingma2014vae]: Kingma, Diederik P and Welling, Max, "Auto-Encoding Variational Bayes", International Conference on Learning Representations
[^rezende2015normalizing]: Rezende, Danilo and Mohamed, Shakir, "Variational Inference with Normalizing Flows", International Conference on Machine Learning
[^lipman2023flow]: Lipman, Yaron and Chen, Ricky T Q and Ben-Hamu, Heli and Nickel, Maximilian and Le, Matthew, "Flow Matching for Generative Modeling", International Conference on Learning Representations
[^goodfellow2014generative]: Goodfellow, Ian and others, "Generative Adversarial Nets", Advances in Neural Information Processing Systems
[^oord2017vqvae]: van den Oord, A{\"a}ron and Vinyals, Oriol and Kavukcuoglu, Koray, "Neural Discrete Representation Learning", Advances in Neural Information Processing Systems
[^defossez2022encodec]: D{\'e}fossez, Alexandre and Copet, Jade and Synnaeve, Gabriel and Adi, Yossi, "High Fidelity Neural Audio Compression", Transactions on Machine Learning Research
[^baevski2020wav2vec]: Baevski, Alexei and Zhou, Yuhao and Mohamed, Abdelrahman and Auli, Michael, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations", Advances in Neural Information Processing Systems
[^hsu2021hubert]: Hsu, Wei-Ning and Bolte, Benjamin and Tsai, Yao-Hung Hubert and Lakhotia, Kushal and Salakhutdinov, Ruslan and Mohamed, Abdelrahman, "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units", IEEE/ACM Transactions on Audio, Speech, and Language Processing
[^chen2022wavlm]: Chen, Sanyuan and Wang, Chengyi and others, "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing", IEEE Journal of Selected Topics in Signal Processing
[^vaswani2017attention]: Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia, "Attention Is All You Need", Advances in Neural Information Processing Systems
[^radford2023robust]: Radford, Alec and Kim, Jong Wook and Xu, Tao and Brockman, Greg and McLeavey, Christine and Sutskever, Ilya, "Robust Speech Recognition via Large-Scale Weak Supervision", International Conference on Machine Learning
[^gulati2020conformer]: Gulati, Anmol and Qin, James and Chiu, Chung-Cheng and others, "Conformer: Convolution-augmented Transformer for Speech Recognition", Interspeech
[^nvidia2024canary]: {NVIDIA}, "Canary: Multi-Lingual Multi-Task ASR and Translation Model", NVIDIA NeMo Toolkit Documentation
[^kim2021conditional]: Kim, Jaehyeon and Kong, Jungil and Son, Juhee, "Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech", International Conference on Machine Learning
[^wang2023valle]: Wang, Chengyi and Chen, Sanyuan and Wu, Yu and others, "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers", arXiv preprint arXiv:2301.02111
[^chen2024f5tts]: Chen, Yushen and others, "F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching", arXiv preprint arXiv:2410.06885
[^du2024cosyvoice]: Du, Zhihao and Chen, Qian and Zhang, Shiliang and others, "CosyVoice: A Scalable Multilingual Zero-shot Text-to-Speech Synthesizer Based on Supervised Semantic Tokens", arXiv preprint arXiv:2407.05407
[^kumar2024dac]: Kumar, Rithesh and Seetharaman, Prem and others, "High-Fidelity Audio Compression with Improved RVQGAN", Advances in Neural Information Processing Systems
[^defossez2024mimi]: D{\'e}fossez, Alexandre and others, "Mimi: A Low-Latency Streaming Audio Codec", arXiv preprint arXiv:2410.00037
[^borsos2023audiolm]: Borsos, Zal{\'a}n and Marinier, Rapha{\"e}l and others, "AudioLM: A Language Modeling Approach to Audio Generation", IEEE/ACM Transactions on Audio, Speech, and Language Processing
[^chu2023qwen2audio]: Chu, Yunfei and Xu, Jin and Zhou, Xiaohuan and others, "Qwen-Audio: Advancing Universal Audio Understanding via Unified Large-Scale Audio-Language Models", arXiv preprint arXiv:2311.07919
[^defossez2024moshi]: D{\'e}fossez, Alexandre and Musicant, Laurent and others, "Moshi: A Speech-Text Foundation Model for Real-Time Dialogue", arXiv preprint arXiv:2410.00037
[^xu2025qwen25omni]: Xu, Jin and Chu, Yunfei and others, "Qwen2.5-Omni Technical Report", arXiv preprint arXiv:2503.20215
[^team2024gemini]: {Gemini Team}, "Gemini: A Family of Highly Capable Multimodal Models", arXiv preprint arXiv:2312.11805
[^communication2023seamlessm4t]: {Seamless Communication}, "SeamlessM4T: Massively Multilingual and Multimodal Machine Translation", arXiv preprint arXiv:2308.11596
[^nguyen2024phowhisper]: Nguyen, Thanh-Nhi and Nguyen, Dat Quoc, "PhoWhisper: Fine-tuning Whisper for Vietnamese Automatic Speech Recognition", arXiv preprint
[^nguyen2020phobert]: Nguyen, Dat Quoc and Nguyen, Anh Tuan, "PhoBERT: Pre-trained Language Models for Vietnamese", Findings of the Association for Computational Linguistics: EMNLP
[^jurafsky2024speech]: Jurafsky, Daniel and Martin, James H, "Speech and Language Processing"
[^huang2001spoken]: Huang, Xuedong and Acero, Alex and Hon, Hsiao-Wuen, "Spoken Language Processing: A Guide to Theory, Algorithm, and System Development"
[^zhang2022wenet]: Zhang, Binbin and Wu, Di and Peng, Zhendong and others, "WeNet 2.0: More Productive End-to-End Speech Recognition Toolkit", Interspeech
[^watanabe2018espnet]: Watanabe, Shinji and Hori, Takaaki and others, "ESPnet: End-to-End Speech Processing Toolkit", Interspeech
[^kuchaiev2019nemo]: Kuchaiev, Oleksii and Li, Jason and others, "NeMo: A Toolkit for Building AI Applications Using Neural Modules", arXiv preprint arXiv:1909.09577
[^ravanelli2021speechbrain]: Ravanelli, Mirco and Parcollet, Titouan and others, "SpeechBrain: A General-Purpose Speech Toolkit", arXiv preprint arXiv:2106.04624
