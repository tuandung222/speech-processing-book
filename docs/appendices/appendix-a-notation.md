# Notation Reference

## General Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x}$ | Waveform signal (1D time-domain) |
| $\mathbf{X}$ | Matrix (uppercase bold) |
| $x_n$ | Sample at time index $n$ |
| $N$ | Number of samples / sequence length |
| $f_s$ | Sampling rate (Hz) |
| $T$ | Number of time frames (mel/feature level) |
| $U$ | Number of text/phoneme tokens |
| $B$ | Batch size |
| $d$ | Model dimension / hidden size |
| $K$ | Codebook size / number of classes |
| $Q$ | Number of RVQ codebook layers |
| $\mathcal{L}$ | Loss function |

## Signal Processing

| Symbol | Meaning | Reference |
|--------|---------|-----------|
| $f_s$ | Sampling frequency (Hz) | Ch. 2 |
| $f_{\text{Nyquist}}$ | Nyquist frequency $= f_s / 2$ | Ch. 2 |
| $X[k]$ | DFT coefficient at frequency bin $k$ | Ch. 2 |
| $w[n]$ | Window function (Hann, Hamming) | Ch. 2 |
| $\text{STFT}(m, k)$ | Short-Time Fourier Transform | Ch. 2 |
| $S_{\text{mel}}$ | Mel spectrogram | Ch. 2 |
| $\text{MFCC}$ | Mel-Frequency Cepstral Coefficients | Ch. 2 |
| $F_0$ / $f_0$ | Fundamental frequency (pitch) | Ch. 2, 10 |
| $\mathcal{M}(f)$ | Mel scale: $2595 \log_{10}(1 + f/700)$ | Ch. 2 |

## ASR Notation

| Symbol | Meaning | Reference |
|--------|---------|-----------|
| $P(\mathbf{y} \mid \mathbf{x})$ | ASR posterior probability | Ch. 3 |
| $\pi$ | CTC alignment path (with blanks) | Ch. 3 |
| $\alpha_t(s)$ | CTC forward variable | Ch. 3 |
| $\beta_t(s)$ | CTC backward variable | Ch. 3 |
| $\text{blank}$ / $\varnothing$ | CTC blank token | Ch. 3 |
| $\alpha_{i,j}$ | Attention weights | Ch. 3, 5 |
| $\text{WER}$ | Word Error Rate | Ch. 3 |
| $\text{CER}$ | Character Error Rate | Ch. 3 |

## Self-Supervised Learning

| Symbol | Meaning | Reference |
|--------|---------|-----------|
| $\mathbf{z}$ | Latent representation | Ch. 4 |
| $\mathbf{q}$ | Quantized representation | Ch. 4 |
| $\mathcal{C}$ | Codebook (set of entries) | Ch. 4, 8 |
| $\mathbf{e}_k$ | Codebook entry $k$ | Ch. 4, 8 |
| $G$ / $V$ | Number of codebook groups / entries | Ch. 4 |
| $\tau$ | Gumbel-Softmax temperature | Ch. 4 |
| $\mathcal{L}_{\text{contrastive}}$ | Contrastive loss | Ch. 4 |
| $\mathcal{L}_{\text{diversity}}$ | Diversity loss | Ch. 4 |

## TTS Notation

| Symbol | Meaning | Reference |
|--------|---------|-----------|
| $\hat{\mathbf{m}}_i$ | Predicted mel frame at step $i$ | Ch. 6 |
| $d_u$ | Duration of phoneme $u$ (in frames) | Ch. 6 |
| $\hat{f}_0(t)$ | Predicted pitch at frame $t$ | Ch. 6 |
| $\hat{e}(t)$ | Predicted energy at frame $t$ | Ch. 6 |
| $p_{\text{stop}}$ | Stop token probability | Ch. 6 |
| $v_\theta$ | Flow matching velocity field | Ch. 7 |
| $\mathbf{x}_t$ | Interpolated sample at flow time $t$ | Ch. 7 |

## Neural Codec Notation

| Symbol | Meaning | Reference |
|--------|---------|-----------|
| $q(\mathbf{z})$ | Vector quantization operator | Ch. 8 |
| $\text{sg}(\cdot)$ | Stop-gradient operator | Ch. 8 |
| $\mathbf{r}_q$ | Residual after $q$-th quantization | Ch. 8 |
| $c_t^{(q)}$ | Codec token at time $t$, codebook $q$ | Ch. 8, 9 |
| $\hat{\mathbf{z}}$ | Reconstructed latent (sum of RVQ) | Ch. 8 |

## Speech LLM Notation

| Symbol | Meaning | Reference |
|--------|---------|-----------|
| $\mathbf{s}$ | Semantic tokens | Ch. 9 |
| $\mathbf{a}^{(q)}$ | Acoustic tokens, codebook $q$ | Ch. 9 |
| $a_t^{\text{user}}$ | User audio token at time $t$ | Ch. 9 |
| $a_t^{\text{AI}}$ | AI audio token at time $t$ | Ch. 9 |
| $w_t$ | Text (inner monologue) token | Ch. 9 |

## Common Abbreviations

| Abbreviation | Full Name |
|-------------|-----------|
| ASR | Automatic Speech Recognition |
| TTS | Text-to-Speech |
| VAD | Voice Activity Detection |
| CTC | Connectionist Temporal Classification |
| RNN-T | Recurrent Neural Network Transducer |
| SSL | Self-Supervised Learning |
| RVQ | Residual Vector Quantization |
| VQ | Vector Quantization |
| STE | Straight-Through Estimator |
| MOS | Mean Opinion Score |
| WER | Word Error Rate |
| RTF | Real-Time Factor |
| G2P | Grapheme-to-Phoneme |
| F0 | Fundamental Frequency |
| MAS | Monotonic Alignment Search |
| CFM | Conditional Flow Matching |
| DiT | Diffusion Transformer |
| MQA | Multi-Query Attention |
