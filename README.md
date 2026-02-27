# finetune-or-prompt

> Should you fine-tune a local model or prompt a frontier API? Now you have empirical data to decide.

Most advice on this question is anecdotal. This project runs the experiments so you don't have to — 50+ fine-tuning runs across 3 tasks, 3 model families, and 4 data sizes, compared against GPT-4o-mini and Claude Haiku with real cost and latency tracking. Results are packaged into a CLI tool that gives you a concrete recommendation based on your actual situation.

```
python recommend.py --task classification --data 300
```

---

## Key Findings

**Task complexity shifts the crossover point.**

| Task | Fine-tune wins when... |
|---|---|
| Binary classification (SST-2) | N ≥ 500 examples |
| Multi-class classification (AG News, 4 classes) | N ≥ 500 for Phi-3, N ≥ 2,000 for Qwen |
| Named entity recognition (WikiANN) | Never, at data sizes tested (up to N=2,000) |

**BERT is criminally underrated.**
BERT-base trains in under 10 seconds, runs at 0.28ms/sample, costs nothing at inference, and matches LoRA fine-tuned Phi-3 mini at N=500 on classification tasks. For classification, it should be your first instinct before reaching for a larger model.

**Fine-tuned models are 50–150x faster at inference than API calls.**
GPT-4o-mini averages 449ms/sample. A fine-tuned Qwen 2.5 1.5B runs at 3ms/sample locally. If you have the data and latency matters, fine-tuning pays for itself quickly.

**Small models are unstable at low data.**
Qwen 2.5 1.5B at N=200 on SST-2 showed accuracy ranging from 0.62 to 0.80 across 3 seeds (std=0.082). Don't trust a single fine-tuning run with fewer than 200 examples.

**Prompting has surprisingly high precision on NER but low recall.**
GPT-4o-mini zero-shot achieves 0.87 precision but only 0.41 recall on WikiANN — it labels conservatively and misses a lot of entities. Fine-tuned models have the opposite problem at low N.

---

## The CLI Tool

### Install

```bash
pip install rich
```

### Usage

```bash
python tool/recommend.py --task TASK --data N [--latency MS] [--cost-sensitive]
```

**Arguments:**

| Argument | Description |
|---|---|
| `--task` | Task type (see below) |
| `--data` | Number of labeled training examples you have |
| `--latency` | Maximum acceptable latency per sample in ms (default: 2000) |
| `--cost-sensitive` | Exclude higher-cost few-shot API options |

**Task options:**

| Alias | Task |
|---|---|
| `classification`, `sentiment`, `binary` | Binary classification |
| `multiclass`, `topic`, `news` | Multi-class classification |
| `ner`, `named-entity`, `tagging` | Named entity recognition |

### Examples

```bash
# Binary classification with 300 examples
python tool/recommend.py --task classification --data 300

# Multi-class with 500 examples and strict latency requirement
python tool/recommend.py --task multiclass --data 500 --latency 100

# NER with 2000 examples, latency budget of 500ms
python tool/recommend.py --task ner --data 2000 --latency 500

# On a budget — exclude expensive few-shot options
python tool/recommend.py --task sentiment --data 50 --cost-sensitive
```

---

## Experimental Setup

### Models

| Model | Type | Parameters | Approach |
|---|---|---|---|
| Qwen 2.5 1.5B | Generative LM | 1.5B | LoRA fine-tune (r=8) |
| Phi-3 mini | Generative LM | 3.8B | LoRA fine-tune (r=8) |
| BERT-base-uncased | Encoder | 110M | Full fine-tune |
| GPT-4o-mini | Frontier API | — | Zero-shot + few-shot prompting |
| Claude Haiku 3 | Frontier API | — | Zero-shot + few-shot prompting |

### Tasks and Datasets

| Task | Dataset | Metric | Classes |
|---|---|---|---|
| Binary classification | SST-2 (GLUE) | Accuracy | 2 |
| Multi-class classification | AG News | Accuracy | 4 |
| Named entity recognition | WikiANN (en) | F1 | 7 (BIO tags) |

### Training Sizes

N = 50, 200, 500, 2,000 examples

### Metrics Tracked

- **Accuracy / F1** — task performance
- **Latency** — ms per sample at inference time
- **Cost per 1,000 inferences** — actual USD for API calls, ~$0 for local models
- **Training time** — wall-clock seconds
- **Variance** — 3 seeds at N=200 and N=500 for classification tasks

### Hardware

Fine-tuning experiments run on NVIDIA RTX 4090 (24GB) and NVIDIA A100 80GB PCIe via RunPod. Total compute cost: ~$25.

---

## Full Results

### SST-2 Binary Classification

| Model | N=50 | N=200 | N=500 | N=2000 | Latency | Cost/1k |
|---|---|---|---|---|---|---|
| Qwen 2.5 1.5B (LoRA) | 59.7% | 80.3% ±8.2 | 83.8% ±0.7 | 94.8% | 3.0ms | ~$0 |
| Phi-3 mini (LoRA) | 60.9% | 89.0% ±2.6 | 92.9% ±0.5 | 94.5% | 8.8ms | ~$0 |
| BERT-base | 51.2% | 75.7% | 85.2% | 89.2% | 0.28ms | ~$0 |
| GPT-4o-mini (0-shot) | — | — | 92.6% | — | 449ms | $0.011 |
| GPT-4o-mini (few-shot) | — | — | 95.4% | — | 493ms | $0.038 |
| Claude Haiku (0-shot) | — | — | 92.6% | — | 676ms | $0.023 |
| Claude Haiku (few-shot) | — | — | 96.2% | — | 720ms | $0.074 |

### AG News Multi-class Classification

| Model | N=50 | N=200 | N=500 | N=2000 | Latency | Cost/1k |
|---|---|---|---|---|---|---|
| Qwen 2.5 1.5B (LoRA) | 48.6% | 72.5% ±3.8 | 81.9% ±3.3 | 90.4% | 2.8ms | ~$0 |
| Phi-3 mini (LoRA) | 70.8% | 83.6% ±0.8 | 88.6% ±0.3 | 91.1% | 7.2ms | ~$0 |
| BERT-base | 48.6% | 76.7% | 85.8% | 89.7% | 0.27ms | ~$0 |
| GPT-4o-mini (0-shot) | — | — | 84.4% | — | 418ms | $0.016 |
| GPT-4o-mini (few-shot) | — | — | 88.4% | — | 454ms | $0.075 |
| Claude Haiku (0-shot) | — | — | 81.0% | — | 673ms | $0.033 |
| Claude Haiku (few-shot) | — | — | 86.0% | — | 777ms | $0.142 |

### WikiANN Named Entity Recognition

| Model | N=50 | N=200 | N=500 | N=2000 | Latency | Cost/1k |
|---|---|---|---|---|---|---|
| Qwen 2.5 1.5B (LoRA) | 0.025 | 0.103 | 0.156 | 0.384 | 3.1ms | ~$0 |
| Phi-3 mini (LoRA) | 0.050 | 0.129 | 0.329 | 0.474 | 8.3ms | ~$0 |
| GPT-4o-mini (0-shot) | — | — | 0.561 | — | 1077ms | $0.038 |
| GPT-4o-mini (few-shot) | — | — | 0.593 | — | 1129ms | $0.054 |
| Claude Haiku (0-shot) | — | — | 0.516 | — | 841ms | $0.080 |
| Claude Haiku (few-shot) | — | — | 0.544 | — | 887ms | $0.114 |

---

## Repo Structure

```
finetune-or-prompt/
├── tool/
│   └── recommend.py        # CLI recommendation tool
├── data/
│   ├── ALL_RESULTS.json          # Raw results from fine-tuning experiments
│   └── ALL_RESULTS_FINAL.json    # Raw results from additional experiments
├── paper/                  # Manuscript (coming soon)
├── requirements.txt
└── README.md
```

---

## Limitations

- Fine-tuning experiments use LoRA (r=8) on generative LMs. Encoder-only models like BERT are included as a baseline but were not subjected to the same hyperparameter tuning.
- NER results reflect generative LMs with token classification heads, which are architecturally suboptimal for sequence labeling. A proper encoder-based NER model would score significantly higher.
- API latency measurements include network round-trip time via OpenRouter and will vary by provider and region.
- Each fine-tuning condition was run with 3 seeds at N=200 and N=500 only. Single-seed results at N=50 and N=2,000 should be interpreted with appropriate uncertainty.
- Results may not generalize to domain-specific tasks (medical, legal, code) where task-specific pretraining matters more.

---

## Citation

If you use this tool or data in your work:

```bibtex
@misc{bhatt2025finetuneorprompt,
  title   = {Fine-Tune or Prompt? An Empirical Decision Framework for NLP Practitioners},
  author  = {Bhatt, Aamod},
  year    = {2025},
  url     = {https://github.com/aamodbhatt/finetune-or-prompt}
}
```

---

## License

MIT
