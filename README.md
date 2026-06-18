# bapX-v1

**Recursive Language Continuous Diffusion with Contextual AutoRegressive Decoder**

---

## 🎯 Overview

**bapX-v1** is a pure Python byte-native diffusion/retrieval model that operates directly on bytes (`0–255`) from local files.

**NO external APIs. NO numpy. NO torch. NO gradient training.**

**RLCoDAR** is the mechanism:

- **RL** = **R**ecursive **L**anguage: files are loaded as context/weights.
- **CoDAR** = **Co**ntinuous **D**iffusion with Contextual **A**uto**R**egressive Decoder.
- **BlockDiffusionConfig** adds DiffusionGemma-style canvas decoding controls while keeping the system byte-native.
- **MaskedByteDiffusionObjective** adds the missing objective layer: corrupt bytes, recover bytes, score reconstruction, then let HyperAgents improve the index/routing.

---

## 🏗️ Architecture

```text
bapX-v1
    │
    └── RLCoDAR
            │
            ├── RL / RLM
            │   ├── load_context()       - Load files as context/weights
            │   ├── REPL environment     - Execute local code
            │   └── Byte indexing        - Index bytes from files
            │
            ├── CoDAR Runtime
            │   ├── ByteIndex            - Searchable byte index
            │   ├── CosineNoiseSchedule  - Diffusion noise schedule
            │   ├── BlockDiffusionConfig - 256-position denoising canvas
            │   ├── accept_canvas()      - Low-entropy byte acceptance
            │   ├── renoise_unaccepted() - Re-noise unresolved positions
            │   └── AR Decoder           - Contextual byte/text output
            │
            └── Objective Harness
                ├── ByteMaskingConfig
                ├── corrupt_bytes()
                ├── recover_bytes()
                ├── train_step()
                └── self_improve_step()
```

This project is **not Gemma** and does not ship Gemma weights. The current update borrows the useful runtime shape from DiffusionGemma: block canvas, denoising budget, entropy acceptance, bidirectional canvas mixing, prompt prefill cache, and re-noising.

---

## 🚀 Quick Start

```bash
# Optional: only needed for HuggingFace dataset streaming
pip install datasets
```

```python
from rlcodar_hyperagi.diffusion import CoDARDiffusion, ByteIndex

byte_index = ByteIndex()
byte_index.add_text("doc1", "Python is a programming language")
byte_index.add_text("doc2", "CoDAR is a byte-native diffusion model")

bapx = CoDARDiffusion(byte_index=byte_index)
response = bapx.reason("What is Python?")
print(response)
```

---

## 🧩 DiffusionGemma-Style Runtime Knobs

The repo now exposes the runtime controls we compared against DiffusionGemma:

```python
from rlcodar_hyperagi.diffusion import (
    ByteIndex,
    CoDARDiffusion,
    BlockDiffusionConfig,
)

byte_index = ByteIndex()
byte_index.add_text("notes", "Byte-native CoDAR uses files as memory.")

config = BlockDiffusionConfig(
    canvas_length=256,
    max_denoising_steps=48,
    entropy_bound=0.1,
    bidirectional_canvas_attention=True,
    encoder_prefill_cache=True,
    accept_low_entropy_tokens=True,
    renoise_unaccepted_tokens=True,
)

model = CoDARDiffusion(byte_index=byte_index, config=config)
print(model.reason("Explain the indexed notes"))
```

| Field | Purpose |
|---|---|
| `canvas_length = 256` | Creates a 256-position byte canvas before decoding. |
| `max_denoising_steps = 48` | Limits the denoising budget to 48 reverse steps. |
| `entropy_bound = 0.1` | Accepts byte positions only when entropy is low enough. |
| `bidirectional_canvas_attention = True` | Lets each canvas position mix with the whole canvas and context signal. |
| `encoder_prefill_cache = True` | Caches prompt embeddings so repeated prompts avoid recomputation. |
| `accept_low_entropy_tokens = True` | Commits low-uncertainty byte groups during denoising. |
| `renoise_unaccepted_tokens = True` | Re-noises unresolved canvas positions for another denoising pass. |

---

## 🎯 Masked Byte Diffusion Objective

The repo now includes a pure-Python objective harness in:

```text
rlcodar_hyperagi/objectives.py
```

This adds the missing **different training objective** concept without pretending to be full neural gradient training.

```python
from rlcodar_hyperagi.diffusion import ByteIndex
from rlcodar_hyperagi.objectives import (
    ByteMaskingConfig,
    MaskedByteDiffusionObjective,
)

idx = ByteIndex()
idx.add_text("Python uses bytes. CoDAR recovers masked byte positions.", source_name="seed")

objective = MaskedByteDiffusionObjective(
    idx,
    config=ByteMaskingConfig(
        mask_ratio=0.15,
        canvas_length=256,
        entropy_bound=0.1,
        smart_masking=True,
        seed=42,
    ),
)

metrics = objective.train_step("Python uses bytes. CoDAR recovers masked byte positions.")
print(metrics)
```

Objective flow:

```text
original bytes
↓
smart mask / random corrupt selected positions
↓
recover masked bytes using ByteIndex + CoDAR runtime
↓
compare recovered bytes with original bytes
↓
record accuracy + byte MAE
↓
optional self_improve_step() adds weak samples back to the index
```

This is the current bridge between:

```text
Diffusion objective idea
+
byte-native file-as-memory system
```

It is intentionally **not** Torch training. It is an objective harness that HyperAgents can use to improve indexing, routing, source selection, and future model-family byte-weight maps.

---

## 📊 How It Works

### 1. RL — Recursive Language

```python
from rlm import RLM

rlm = RLM(environment="local", persistent=True)
with open("my_code.py") as f:
    rlm._persistent_env.load_context({"code": f.read()})
```

Files become the working context/weight source.

### 2. CoDAR — Byte Diffusion + AR Decoder

```python
from rlcodar_hyperagi.diffusion import CoDARDiffusion, ByteIndex, CosineNoiseSchedule

byte_index = ByteIndex()
byte_index.add_text("file.py", open("file.py").read())

schedule = CosineNoiseSchedule(T=1000)
codar = CoDARDiffusion(byte_index=byte_index, schedule=schedule)
response = codar.reason("What does file.py do?")
```

### 3. Block Canvas Decoding

```text
prompt
↓
encoder prefill embedding cache
↓
256-position byte canvas
↓
reverse diffusion steps
↓
low-entropy accept
↓
re-noise unresolved positions
↓
contextual AR output
```

---

## 🔬 Technical Details

### Why Bytes?

Bytes are the universal substrate of compute. Text, code, images, audio, model weights, GGUF, safetensors, JSON, Python packages, and runtime artifacts all end up as byte streams on disk.

A byte-native model can therefore treat every modality as one common addressable domain:

```text
text/code/image/audio/model files
↓
raw bytes 0-255
↓
ByteIndex / canvas / objective
```

### Byte-Level Tokenization

```python
# "Hello World!" → Byte groups
[[72, 101, 108, 108, 111],    # "Hello"
 [32],                         # " "
 [87, 111, 114, 108, 100, 33]] # "World!"
```

### Diffusion Process

```text
Forward diffusion:
q(x_t | x_0) = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise

Reverse diffusion:
x_{t-1} = x_t + context_guided_velocity + optional_noise
```

### Low-Entropy Acceptance

The block canvas path computes normalized byte entropy for every canvas position. A position is committed when entropy is below `entropy_bound`. Unaccepted positions can be re-noised and denoised again.

---

## 📁 File Structure

```text
RecursiveLM/
├── rlm/                           # RLM file/context layer
│   ├── core/
│   ├── environments/
│   └── datasets/
│
├── rlcodar_hyperagi/
│   ├── diffusion.py               # CoDAR runtime
│   │   ├── ByteIndex
│   │   ├── ByteGroupTokenizer
│   │   ├── CosineNoiseSchedule
│   │   ├── BlockDiffusionConfig
│   │   └── CoDARDiffusion
│   └── objectives.py              # Masked byte diffusion objective
│       ├── ByteMaskingConfig
│       └── MaskedByteDiffusionObjective
│
└── hyperagents/                   # HyperAgents / self-improvement layer
```

---

## 🧪 Testing

```bash
python rlcodar_hyperagi/diffusion.py
python rlcodar_hyperagi/objectives.py
```

Expected verification areas:

- byte-group tokenization
- byte index search
- cosine noise schedule
- block diffusion canvas config
- prompt prefill cache
- low-entropy accept/re-noise path
- masked byte corruption/recovery objective
- OpenAI-compatible `completion()` method

---

## 📊 Performance Notes

| Metric | Current Direction |
|---|---|
| Parameters | 0 trained neural parameters; files act as context/weights |
| Vocabulary | 256 byte values |
| Context | File/index driven |
| Runtime | Pure Python |
| Block Canvas | 256 positions |
| Denoising Budget | 48 steps by default |
| Objective | Masked byte recovery, non-gradient |

---

## 🎯 Use Cases

### Code Understanding

```python
byte_index = ByteIndex()
byte_index.add_directory("./my-project", extensions=[".py", ".md", ".json"])

bapx = CoDARDiffusion(byte_index=byte_index)
print(bapx.reason("How does authentication work?"))
```

### Documentation Q&A

```python
byte_index.add_text("README.md", open("README.md").read())
bapx = CoDARDiffusion(byte_index=byte_index)
print(bapx.reason("What is the API for diffusion?"))
```

### Dataset Q&A

```python
from rlm.datasets import DatasetStreamer

streamer = DatasetStreamer("fineweb", limit=100)
for i, doc in enumerate(streamer.stream_fineweb()):
    byte_index.add_text(f"fineweb_{i}", doc["text"])
```

---

## 📚 References / Related Ideas

- Recursive Language Models / file-as-context systems
- Continuous diffusion with contextual AR decoding
- Masked discrete diffusion language models
- DiffusionGemma-style block diffusion controls
- HyperAgents / self-improving agent loops

---

## 📄 License

MIT License - See LICENSE file for details.

---

**bapX-v1: Pure Python byte-native diffusion. Your files are the working memory. No external API required.**
