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

Cuốn sách gồm **20 chương** chia thành **7 phần** và **6 phụ lục**:

```
Speech AI: From Signal Processing to Full-Duplex Dialogue
│
├── Phần I: Nền tảng (Ch 1-3)
│   ├── Ch 1: From NLP to Speech - The Conceptual Bridge
│   ├── Ch 2: Audio Signal Fundamentals (DFT, STFT, Mel, MFCC)
│   └── Ch 3: Speech Representations (SSL, Contrastive, Codecs)
│
├── Phần II: Nhận dạng Giọng nói - ASR (Ch 4-7)
│   ├── Ch 4: ASR Foundations (CTC, Seq2Seq, RNN-T)
│   ├── Ch 5: Modern ASR Architectures (Conformer, Zipformer, E-Branchformer)
│   ├── Ch 6: Whisper & Canary
│   └── Ch 7: Streaming ASR
│
├── Phần III: Tổng hợp Giọng nói - TTS & Audio Codecs (Ch 8-10)
│   ├── Ch 8: TTS Foundations & Vocoders
│   ├── Ch 9: End-to-End TTS & Zero-Shot Voice Cloning
│   └── Ch 10: Audio Codecs & Neural Tokenization
│
├── Phần IV: Speech LLMs & Multimodal (Ch 11-13)
│   ├── Ch 11: Speech Language Models (AudioLM, Qwen2-Audio, Moshi)
│   ├── Ch 12: Multimodal Omni Models (Qwen2.5-Omni, Gemini, GPT-4o)
│   └── Ch 13: Full-Duplex Dialogue
│
├── Phần V: Speech Understanding & Translation (Ch 14-15)
│   ├── Ch 14: Speech Classification (SER, SID, LID, AED)
│   └── Ch 15: Speech Translation (S2ST, S2TT, SeamlessM4T)
│
├── Phần VI: Tiếng Việt (Ch 16-17)
│   ├── Ch 16: Vietnamese Speech Processing
│   └── Ch 17: Vietnamese Datasets & Benchmarks
│
├── Phần VII: Tools & Production (Ch 18-20)
│   ├── Ch 18: Training Frameworks (WeNet, ESPnet, NeMo, SpeechBrain)
│   ├── Ch 19: Inference Engines (TensorRT, Triton, ONNX, CTranslate2)
│   └── Ch 20: Production Speech Systems
│
└── Phụ lục A-F
    ├── A: Notation Reference
    ├── B: Mathematical Proofs
    ├── C: NLP-Speech Mapping
    ├── D: Code Listings
    ├── E: Vietnamese Speech Resources
    └── F: Tool Comparison Matrices
```

## Hướng dẫn Đọc

### Đến từ LLM/GPT research:

1. **Chương 1** (NLP-to-Speech Bridge) - bắt đầu từ đây
2. **Chương 11-13** (Speech LLMs, Multimodal, Full-Duplex) - lãnh thổ quen thuộc nhất
3. **Chương 2-3** (Audio Fundamentals, Representations) - bổ sung signal processing
4. Các chương còn lại theo thứ tự

### Đến từ NLP/BERT research:

1. **Chương 1** - **Chương 2-3** - **Chương 4** (ASR ~ seq2seq translation)
2. **Chương 5-7** (Modern ASR) - **Chương 8-10** (TTS & Codecs) - **Chương 11-13** (Speech LLMs)

### Đến từ CV/Computer Vision:

1. **Chương 2** (signal processing tuong tu image features)
2. **Chương 14** (speech classification ~ image classification)
3. **Chương 3** (self-supervised learning ~ SimCLR/DINO for audio)

### Muốn triển khai production ngay:

1. **Chương 18-20** (Training frameworks, Inference engines, Production systems)
2. **Phụ lục F** (Tool comparison matrices)
3. Quay lại lý thuyết khi cần

### Đọc toàn bộ (comprehensive):

Chương 1 - 2 - 3 - ... - 20, theo thứ tự. Đây là lộ trình được recommend.

## Conventions

Xuyên suốt cuốn sách, chúng tôi sử dụng các conventions sau:

!!! note "Intuition"
    Giải thích trực giác về một khái niệm phức tạp.


!!! warning "Hardware/Latency Warning"
    Cảnh báo về performance, latency, memory bottleneck, hoặc hardware constraints trong production.


!!! tip "NLP Parallel"
    Mapping giữa khái niệm NLP quen thuộc và speech equivalent.


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
