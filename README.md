# finetune-or-prompt

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License">
  <img src="https://img.shields.io/badge/arXiv-2502.XXXXX-b31b1b.svg" alt="arXiv">
  <img src="https://img.shields.io/badge/experiments-50%2B%20runs-orange.svg" alt="50+ experiments">
</p>

<p align="center">
  <b>Should you fine-tune a local model or prompt a frontier API?</b><br>
  We ran 50+ experiments across 3 tasks, 3 model families, and 4 data sizes so you don't have to.
</p>

---

## Overview

Most advice on fine-tuning vs. prompting is anecdotal. This project runs the experiments systematically — comparing LoRA fine-tuning of small local models against GPT-4o-mini and Claude Haiku 3, with **cost and latency tracked as first-class metrics**.

Results are packaged into a CLI tool that gives you a concrete recommendation based on your task, data size, latency budget, and cost constraints.
```bash
python tool/recommend.py --task classification --data 500
```
```
╭────────────────────────────────────────────────╮
│ finetune-or-prompt  ·  empirical decision tool │
╰────────────────────────────────────────────────╯

Recommendation:
✅ Fine-tune  ·  Phi-3-mini

   ACCURACY     0.929
   Latency      8.8ms / sample
   Cost/1k inf  ~$0 (local)
   Train time   70s
```

---

## Key Findings

> **Task complexity shifts the crossover point.** There is no universal answer — the right choice depends on your task.

| Task | Fine-tune wins when... | Prompting wins when... |
|---|---|---|
| Binary classification | N ≥ 500 examples | N < 500 |
| Multi-class classification (4 classes) | N ≥ 500 (Phi-3), N ≥ 2,000 (Qwen) | N < 500 |
| Named entity recognition | Never (at N ≤ 2,000) | Always |

**Other findings:**

- **BERT is criminally underrated.** BERT-base trains in under 10 seconds, runs at 0.28ms/sample, and matches LoRA fine-tuned Phi-3 mini on classification at N=500. It should be your first instinct before reaching for a larger architecture.
- **Fine-tuned models are 50–150x faster at inference.** GPT-4o-mini averages 449ms/sample. Phi-3 mini fine-tuned locally runs at 8.8ms/sample. At scale, this is the difference between a responsive and unresponsive system.
- **Low-data fine-tuning is unreliable.** Qwen 2.5 1.5B at N=200 showed accuracy ranging from 62% to 80% across seeds (std=8.2%). Never trust a single fine-tuning run below N=200.
- **API models are precise but miss entities.** GPT-4o-mini on NER: precision=0.87, recall=0.41. They label conservatively and miss over half of all entities.

---

## Results

### Binary Classification — SST-2

| Model | N=50 | N=200 | N=500 | N=2000 | Latency | Cost/1k |
|---|---|---|---|---|---|---|
| Qwen 2.5 1.5B (LoRA) | 59.7% | 80.3% ±8.2 | 83.8% ±0.7 | 94.8% | 3.0ms | ~$0 |
| Phi-3 mini (LoRA) | 60.9% | 89.0% ±2.6 | 92.9% ±0.5 | 94.5% | 8.8ms | ~$0 |
| BERT-base | 51.2% | 75.7% | 85.2% | 89.2% | 0.28ms | ~$0 |
| GPT-4o-mini (0-shot) | — | — | 92.6% | — | 449ms | $0.011 |
| GPT-4o-mini (few-shot) | — | — | 95.4% | — | 493ms | $0.038 |
| Claude Haiku 3 (0-shot) | — | — | 92.6% | — | 676ms | $0.023 |
| Claude Haiku 3 (few-shot) | — | — | 96.2% | — | 720ms | $0.074 |

### Multi-class Classification — AG News (4 classes)

| Model | N=50 | N=200 | N=500 | N=2000 | Latency | Cost/1k |
|---|---|---|---|---|---|---|
| Qwen 2.5 1.5B (LoRA) | 48.6% | 72.5% ±3.8 | 81.9% ±3.3 | 90.4% | 2.8ms | ~$0 |
| Phi-3 mini (LoRA) | 70.8% | 83.6% ±0.8 | 88.6% ±0.3 | 91.1% | 7.2ms | ~$0 |
| BERT-base | 48.6% | 76.7% | 85.8% | 89.7% | 0.27ms | ~$0 |
| GPT-4o-mini (0-shot) | — | — | 84.4% | — | 418ms | $0.016 |
| GPT-4o-mini (few-shot) | — | — | 88.4% | — | 454ms | $0.075 |
| Claude Haiku 3 (0-shot) | — | — | 81.0% | — | 673ms | $0.033 |
| Claude Haiku 3 (few-shot) | — | — | 86.0% | — | 777ms | $0.142 |

### Named Entity Recognition — WikiANN

| Model | N=50 | N=200 | N=500 | N=2000 | Latency | Cost/1k |
|---|---|---|---|---|---|---|
| Qwen 2.5 1.5B (LoRA) | 0.025 | 0.103 | 0.156 | 0.384 | 3.1ms | ~$0 |
| Phi-3 mini (LoRA) | 0.050 | 0.129 | 0.329 | 0.474 | 8.3ms | ~$0 |
| GPT-4o-mini (0-shot) | — | — | 0.561 | — | 1,077ms | $0.038 |
| GPT-4o-mini (few-shot) | — | — | 0.593 | — | 1,129ms | $0.054 |
| Claude Haiku 3 (0-shot) | — | — | 0.516 | — | 841ms | $0.080 |
| Claude Haiku 3 (few-shot) | — | — | 0.544 | — | 887ms | $0.114 |

---

## CLI Tool

### Installation
```bash
git clone https://github.com/YOUR_USERNAME/finetune-or-prompt
cd finetune-or-prompt
pip install rich
```

### Usage
```bash
python tool/recommend.py --task TASK --data N [--latency MS] [--cost-sensitive]
```

| Argument | Description |
|---|---|
| `--task` | Task type (see aliases below) |
| `--data` | Number of labeled training examples you have |
| `--latency` | Max acceptable inference latency in ms (default: 2000) |
| `--cost-sensitive` | Exclude higher-cost few-shot API options |

**Task aliases:**

| Input | Task |
|---|---|
| `classification`, `sentiment`, `binary` | Binary classification |
| `multiclass`, `topic`, `news` | Multi-class classification |
| `ner`, `named-entity`, `tagging` | Named entity recognition |

### Examples
```bash
# Do I have enough data to fine-tune for sentiment analysis?
python tool/recommend.py --task sentiment --data 300

# Multi-class with a strict latency requirement
python tool/recommend.py --task multiclass --data 500 --latency 100

# NER on a budget — no expensive few-shot calls
python tool/recommend.py --task ner --data 2000 --cost-sensitive

# Very low data, want cheapest reliable option
python tool/recommend.py --task classification --data 50 --cost-sensitive
```

---

## Experimental Setup

### Models

| Model | Type | Params | Approach |
|---|---|---|---|
| Qwen 2.5 1.5B | Generative LM | 1.5B | LoRA (r=8, α=16) |
| Phi-3 mini | Generative LM | 3.8B | LoRA (r=8, α=16) |
| BERT-base-uncased | Encoder | 110M | Full fine-tune |
| GPT-4o-mini | Frontier API | — | Zero-shot + few-shot |
| Claude Haiku 3 | Frontier API | — | Zero-shot + few-shot |

### Training Details

| Setting | Value |
|---|---|
| LoRA rank / alpha | r=8, α=16 |
| LoRA dropout | 0.1 |
| LoRA target modules | q_proj, v_proj (Qwen) · qkv_proj, o_proj (Phi-3) |
| Precision | BF16 (LoRA) · FP16 (BERT) |
| Learning rate | 2e-4 (LoRA) · 2e-5 (BERT) |
| Epochs | 10 / 6 / 4 / 3 for N = 50 / 200 / 500 / 2,000 |
| Seeds | 3 seeds (0, 1, 42) at N=200 and N=500 |
| Hardware | NVIDIA RTX 4090 24GB · A100 80GB PCIe (RunPod) |
| Total compute cost | ~$25 USD |

---

## Repo Structure
```
finetune-or-prompt/
├── tool/
│   └── recommend.py               # CLI recommendation tool
├── data/
│   ├── ALL_RESULTS.json           # SST-2 and NER results
│   └── ALL_RESULTS_FINAL.json     # AG News, BERT, seed variance, 500-example API evals
├── paper/
│   └── paper_academic.docx        # Full paper (also on arXiv)
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Paper

> **Fine-Tune or Prompt? An Empirical Decision Framework for NLP Practitioners**
> Aamod Bhatt · cs.CL · February 2025
> [arXiv:2502.XXXXX](https://arxiv.org/abs/2502.XXXXX) [TO BE PUBLISHED]
```bibtex
@misc{bhatt2025finetuneorprompt,
  title   = {Fine-Tune or Prompt? An Empirical Decision Framework for NLP Practitioners},
  author  = {Bhatt, Aamod},
  year    = {2025},
  eprint  = {2502.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```

---

## Limitations

- LoRA experiments use r=8 with default hyperparameters — results reflect practical defaults, not tuned optima
- NER results reflect generative LMs with classification heads, which are architecturally suboptimal for token-level labeling; encoder-based NER models would score substantially higher
- API latency includes network round-trip via OpenRouter and will vary by provider and region
- Single-seed results at N=50 and N=2,000; multi-seed only at the crossover region (N=200, N=500)
- Results may not generalize to domain-specific tasks (medical, legal, scientific)

---

## License

MIT License — Copyright (c) 2025 Aamod Bhatt

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
