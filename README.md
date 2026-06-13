# SLMa-94M — Bare-Metal Edge AI

> A 94M-parameter language model trained from scratch and deployed on a 2 GB Android phone — no PyTorch, no ONNX, no runtime dependencies.

[![Model](https://img.shields.io/badge/🤗%20Model-amartya--pandey%2FSLMa--94M-blue)](https://huggingface.co/amartya-pandey/SLMa-94M)
[![Tokenizer](https://img.shields.io/badge/🤗%20Tokenizer-slm--tokenizer--24k-blue)](https://huggingface.co/amartya-pandey/slm-tokenizer-24k)

---

## What This Is

Most edge AI projects fine-tune existing models. This one builds the entire stack from scratch:

- **Pre-training** — A Llama-style transformer trained on synthetic educational data (Cosmopedia) on a single Kaggle T4 GPU using the "Phi Strategy": maximize reasoning per parameter with highly structured data.
- **Export** — Weights and tokenizer serialized to flat binary files.
- **Inference** — A pure-C engine that `mmap`s those binaries directly into memory, with zero external dependencies.

The result runs on a 2 GB Android phone via Termux.

---

## Architecture

| Hyperparameter | Value |
|---|---|
| Parameters | ~94.3 M |
| Transformer Layers | 12 |
| Embedding Dimension | 768 |
| Context Window | 768 tokens |
| Vocabulary Size | 24,576 (divisible by 128 for Tensor Core alignment) |
| Attention | Grouped-Query Attention — 12 Query heads, 4 KV heads |
| Activations | SwiGLU |
| Normalization | RMSNorm |
| Position Embeddings | RoPE (real-valued) |

---

## Repository Structure

```
SLMa-94M-baremetal/
├── export/
│   ├── export_weights.py      # Serialize safetensors → binaries/model.bin
│   └── export_tokenizer.py    # Serialize HF tokenizer → binaries/tokenizer.bin
├── SLMa-94M/                  # ← clone the HF repo here (see setup below)
│   ├── config.json
│   ├── generation_config.json
│   ├── model.py
│   └── model.safetensors
├── binaries/                  # Pre-built binaries (ready to use)
│   ├── model.bin              # Exported weight binary
│   └── tokenizer.bin          # Exported tokenizer binary
├── run.c                      # Bare-metal C inference engine
└── slm-94m-train-run1.ipynb   # End-to-end training notebook
```

---

## Quick Start

### 1. Clone This Repo

```bash
git clone https://github.com/your-username/SLMa-94M-baremetal
cd SLMa-94M-baremetal
```

### 2. Get the Binaries

**Option A — Use the pre-built binaries (easiest)**

The `binaries/` folder already contains ready-to-use `model.bin` and `tokenizer.bin`. Skip straight to [step 3](#3-compile).

**Option B — Build the binaries yourself**

If you want to regenerate them from the raw weights, first clone the Hugging Face model repo into the root of this repository:

```bash
# Install git-lfs if you haven't already
git lfs install

# Clone the HF repo into the root dir as SLMa-94M/
git clone https://huggingface.co/amartya-pandey/SLMa-94M
```

Then run the export scripts:

```bash
pip install safetensors transformers

python export/export_weights.py       # → binaries/model.bin
python export/export_tokenizer.py     # → binaries/tokenizer.bin
```

### 3. Compile

**Termux / Android / Edge (standard):**
```bash
gcc -O3 -o run run.c -lm
```

**Desktop / Server (with OpenMP for parallel matmul):**
```bash
gcc -O3 -fopenmp -o run run.c -lm
```

### 4. Run

**Text generation:**
```bash
./run binaries/model.bin -z binaries/tokenizer.bin -i "Let's talk about matrix in mathematics " -n 256
```

**Interactive chat:**
```bash
./run binaries/model.bin -z binaries/tokenizer.bin -m chat
```

**CLI flags:**

| Flag | Description | Default |
|---|---|---|
| `-i <string>` | Input prompt | — |
| `-m <string>` | Mode: `generate` or `chat` | `generate` |
| `-n <int>` | Max tokens to generate | 256 |
| `-t <float>` | Temperature | 0.7 |
| `-p <float>` | Top-P (nucleus sampling) | 0.9 |
| `-s <int>` | Random seed | — |

---

## Python Inference (via Hugging Face)

If you prefer to run inference in Python rather than the bare-metal C engine, you can load the model directly from the Hugging Face Hub using `transformers`.

```bash
pip install torch transformers
```

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

repo_id = "amartya-pandey/SLMa-94M"
device = "cuda" if torch.cuda.is_available() else "cpu"

# Always use the custom tokenizer — do not substitute a generic one
tokenizer = AutoTokenizer.from_pretrained("amartya-pandey/slm-tokenizer-24k")
model = AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True).to(device)
model.eval()
```

### Generation function

```python
def generate(prompt, tokenizer, **kwargs):
    defaults = dict(
        max_new_tokens=300,
        min_new_tokens=10,
        early_stopping=True,
        max_time=100.0,
        do_sample=True,
        temperature=0.3,
        top_k=40,
        top_p=0.85,
        typical_p=0.9,
        epsilon_cutoff=0.0003,
        eta_cutoff=0.0003,
        repetition_penalty=1.1,
        length_penalty=1.0,
        no_repeat_ngram_size=3,
        num_beams=2,
        num_beam_groups=1,
        diversity_penalty=0.0,
        guidance_scale=1.0,
        num_return_sequences=1,
        output_scores=False,
        return_dict_in_generate=False,
    )
    defaults.update(kwargs)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, **defaults)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Example usage

**Greedy / deterministic generation:**

```python
print(generate(
    prompt="Let's talk about matrix in mathematics ",
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=False,       # greedy decoding
    num_beams=1,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
))
```

**Sampling with beam search:**

```python
print(generate(
    prompt="Let's talk about matrix in mathematics ",
    tokenizer=tokenizer,
    max_new_tokens=300,
    do_sample=True,
    temperature=0.3,
    top_k=40,
    top_p=0.85,
    num_beams=2,
    repetition_penalty=1.1,
    no_repeat_ngram_size=3,
))
```

---

## Training

The full pipeline lives in `slm-94m-train-run1.ipynb`. Key details:

**Data** — Streams `HuggingFaceTB/cosmopedia` and packs token sequences into block-diagonal matrices via a custom `IsolatedPackedDataset`, preventing cross-document attention leakage.

**Optimization:**

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 1e-3 → 1e-4 (linear warmup + cosine decay) |
| Weight Decay | 0.1 |
| Precision | `torch.autocast` (float16) |
| Gradient Accumulation | 4 steps |
| Hardware | Single Kaggle T4 GPU |
| Experiment Tracking | Weights & Biases |

---

## How the C Engine Works

`run.c` is a self-contained inference engine implementing:

- **Memory mapping** — Model weights are `mmap`'d from `model.bin`, avoiding a full load into RAM.
- **Forward pass** — RMSNorm, RoPE, GQA attention with KV cache, SwiGLU FFN, and a tied output projection.
- **Sampling** — Temperature scaling and Top-P (nucleus) sampling.
- **Tokenization** — BPE decoding read directly from `tokenizer.bin`.

No Python. No PyTorch. No ONNX. The entire runtime fits in a single C file.