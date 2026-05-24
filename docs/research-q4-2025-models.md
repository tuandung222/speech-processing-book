# Research Notes — Q4 2025 / 2026 Speech AI Models

Compiled from Tavily web search during autonomous rewrite session 2.

## Qwen-Omni Family

### Qwen3-Omni (September 2025)
- **Repo**: <https://github.com/QwenLM/Qwen3-Omni>
- **Variants**: Qwen3-Omni-30B-A3B-Instruct, Thinking, Captioner
- **Architecture**: MoE Thinker-Talker design; Thinker for multimodal reasoning, Talker for streaming speech generation, Code2Wav for waveform output
- **Languages**: 119 text input, 19 speech input including Vietnamese, 10 speech output excluding Vietnamese in the official README list
- **Benchmarks**: official report claims leading results on 22/36 audio/video benchmarks and open-source leading results on 32/36; verify per benchmark protocol
- **Comparable to**: Gemini 2.5 Pro on several reported ASR, audio understanding, and voice conversation benchmarks
- **License**: Apache 2.0
- **Reasoning model**: Qwen3-Omni-30B-A3B-Thinking, reasoning-oriented and primarily text-output in official model descriptions
- **Serving**: vLLM-Omni supports `vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct --omni`; stage-based deployment splits Thinker, Talker, and Code2Wav

### Qwen3-Omni-Flash (December 2025)
- Faster variant, optimized latency
- URL: <https://qwen.ai/blog?id=qwen3-omni-flash-20251201>

### Qwen3.5-Omni (March 30, 2026)
- **Plus variant**: official report claims many leading results across audio, audio-video, reasoning, and interaction benchmarks
- Reported favorable comparisons against Gemini 3.1 Pro on selected general audio understanding, reasoning, and translation benchmarks; verify per benchmark protocol
- Native multimodal (text, image, audio, video in single pass)
- Real-time streaming speech output

## OpenAI Voice Stack

### gpt-realtime (August 2025)
- General availability of Realtime API
- Most advanced speech-to-speech model from OpenAI
- New features: MCP server support, image input, SIP phone calling
- Better instruction following, tool calling, natural expressive speech
- URL: <https://openai.com/index/introducing-gpt-realtime>
- Pricing: ~$15-30/month, $0.015-0.03 per minute (Realtime TTS 1)

### Basic Voice Mode retirement
- Sep 9, 2025: Basic Voice Mode fully retired
- Advanced Voice Mode = realtime model

### Realtime translation, Realtime transcription
- New API endpoints for live multilingual voice

## Kyutai (Moshi family)

### Moshi (original, Sep 2024)
- 7.6B params, on-device speech-to-speech
- Mimi codec (12.5 Hz frame rate)
- Theoretical latency 160ms, 200ms in practice
- Open source

### Moshi v2 / MoshiRAG (April 2026)
- Asynchronous knowledge retrieval for full-duplex speech LMs
- Helps Moshi answer tough questions with help from text LLM

### Kyutai TTS / Pocket TTS
- 100M params TTS reported to be competitive with much larger models in the authors' evaluation setup
- Voice cloning support
- Real-time streaming

### MoshiVis
- Image input support for Moshi
- Real-time latency preserved

## Meta SeamlessM4T

### SeamlessM4T (Aug 2023)
- Original: 100 languages, S2TT/S2ST/T2TT/T2ST/ASR

### SeamlessM4T v2 (Nov 2023, Nature 2024 paper)
- 30.1 BLEU on S2TT, 20.5 BLEU on S2ST
- Published in Nature journal (Joint speech and text MT for up to 100 languages)
- 96 target text languages, 36 target speech languages

### SeamlessStreaming
- Simultaneous translation
- EMMA (Efficient Monotonic Multihead Attention)
- 26.8 BLEU S2TT streaming

**Note**: SeamlessM4T v3 not found in current public docs as of search date. Likely still v2.

## Voice Agent TTS providers

### Latency benchmarks (Inworld AI 2026 comparison)
- Realtime TTS (Inworld): Sub-250ms
- ElevenLabs Flash: 75ms
- OpenAI: ~500ms
- Amazon Polly: 100ms-1s

### Pricing (per month subscription)
- ElevenLabs: $60-120
- OpenAI: $15-30
- Amazon Polly: $16-30

### Pay-per-character (cost per 1k chars)
- ElevenLabs Flash: $0.06-0.12
- OpenAI: $0.015-0.03
- Amazon Polly: $0.02-0.03

### Voice agent cost (per minute)
- Typical neural TTS: $0.01-0.02 per minute of generated speech
- Full voice agent (STT + LLM + TTS + orchestration): $0.05-0.20 per minute

## Vietnamese Speech

### PhoWhisper (VinAI, ICLR 2024 Tiny Paper)
- Authors: Thanh-Thien Le, Linh The Nguyen, Dat Quoc Nguyen
- Fine-tuned Whisper on 844 hours Vietnamese ASR
- Datasets: CMV-Vi (14h), VIVOS (14h), VLSP 2020 (240h), VinAI private (586h)
- 5 model sizes (tiny → large)
- SOTA on VLSP 2020 Task-1, Task-2, CMV-Vi, VIVOS benchmarks
- Repo: <https://github.com/VinAIResearch/PhoWhisper>
- HF: <https://huggingface.co/vinai/PhoWhisper-base>

### Vietnamese ASR landscape
- VinBigData/VinAI: ViVi assistant, PhoWhisper, PhoBERT/PhoGPT base
- ZaloAI: KiKi assistant, ZaloPay voice
- FPT.AI: FPT Speech
- VinFast: in-car voice assistant
- Trusting Social: voice biometrics + KYC

## Production Voice Agent Stacks

### Open-source / self-host
- LiveKit (WebRTC infrastructure)
- Pipecat (pipeline framework)
- Vapi.ai (voice agent platform)
- Cartesia Sonic (TTS)
- Moshi (self-host)

### Commercial APIs
- OpenAI Realtime API (gpt-realtime)
- ElevenLabs Voice Agents
- Deepgram (Aura-2 TTS, Nova-3 STT, voice agent bundle)
- Gradium (unified TTS+STT+cloning)
- Cartesia Sonic 2

### Standard latency budget for voice agent
- STT: 200-300ms (streaming)
- LLM first token: 200-500ms
- TTS first byte: 75-300ms
- Total perceived: <1s ideal, <1.5s acceptable

## Wake-word / Keyword Spotting

### Open-source toolkits
- Porcupine (Picovoice): commercial w/ open SDKs
- Snowboy (deprecated, but historically important)
- Mycroft Precise
- Howl

### Industry implementations
- Apple "Hey Siri": 2-stage (small DNN on chip + larger verification on AP)
- Google "OK Google": DS-CNN architecture history, then Conformer-KWS
- Amazon Alexa: small CNN on-device
- Microsoft Cortana: similar architecture

### Datasets
- Google Speech Commands V2 (35 keywords, 105k utterances)
- Hey Snips
- MLCommons Multilingual Spoken Words Corpus
