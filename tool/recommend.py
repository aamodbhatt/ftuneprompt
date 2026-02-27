#!/usr/bin/env python3
"""
finetune-or-prompt: An empirical decision tool for ML practitioners.
Based on experiments comparing LoRA fine-tuning vs prompting across
task types, data sizes, model families, and cost constraints.
"""

import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

# ── Empirical results ─────────────────────────────────────────────────────────
RESULTS = {
    "classification_binary": {
        "fine_tune": {
            "Qwen2.5-1.5B": {
                50:   {"accuracy": 0.597, "latency_ms": 3.0,  "train_time_sec": 31},
                200:  {"accuracy": 0.803, "latency_ms": 3.0,  "train_time_sec": 26},
                500:  {"accuracy": 0.838, "latency_ms": 3.0,  "train_time_sec": 26},
                2000: {"accuracy": 0.948, "latency_ms": 3.0,  "train_time_sec": 55},
            },
            "Phi-3-mini": {
                50:   {"accuracy": 0.609, "latency_ms": 8.8,  "train_time_sec": 87},
                200:  {"accuracy": 0.890, "latency_ms": 8.8,  "train_time_sec": 70},
                500:  {"accuracy": 0.929, "latency_ms": 8.8,  "train_time_sec": 70},
                2000: {"accuracy": 0.945, "latency_ms": 8.8,  "train_time_sec": 139},
            },
            "BERT-base": {
                50:   {"accuracy": 0.512, "latency_ms": 0.28, "train_time_sec": 4},
                200:  {"accuracy": 0.757, "latency_ms": 0.28, "train_time_sec": 4},
                500:  {"accuracy": 0.852, "latency_ms": 0.28, "train_time_sec": 5},
                2000: {"accuracy": 0.892, "latency_ms": 0.28, "train_time_sec": 11},
            },
        },
        "prompt": {
            "GPT-4o-mini": {
                "zero-shot": {"accuracy": 0.926, "latency_ms": 449, "cost_per_1k": 0.011},
                "few-shot":  {"accuracy": 0.954, "latency_ms": 493, "cost_per_1k": 0.038},
            },
            "Claude-Haiku-3": {
                "zero-shot": {"accuracy": 0.926, "latency_ms": 676, "cost_per_1k": 0.023},
                "few-shot":  {"accuracy": 0.962, "latency_ms": 720, "cost_per_1k": 0.074},
            },
        },
    },
    "classification_multiclass": {
        "fine_tune": {
            "Qwen2.5-1.5B": {
                50:   {"accuracy": 0.486, "latency_ms": 2.8,  "train_time_sec": 61},
                200:  {"accuracy": 0.725, "latency_ms": 2.8,  "train_time_sec": 45},
                500:  {"accuracy": 0.819, "latency_ms": 2.8,  "train_time_sec": 42},
                2000: {"accuracy": 0.904, "latency_ms": 2.8,  "train_time_sec": 63},
            },
            "Phi-3-mini": {
                50:   {"accuracy": 0.708, "latency_ms": 7.2,  "train_time_sec": 154},
                200:  {"accuracy": 0.836, "latency_ms": 7.2,  "train_time_sec": 109},
                500:  {"accuracy": 0.886, "latency_ms": 7.2,  "train_time_sec": 95},
                2000: {"accuracy": 0.911, "latency_ms": 7.2,  "train_time_sec": 153},
            },
            "BERT-base": {
                50:   {"accuracy": 0.486, "latency_ms": 0.27, "train_time_sec": 7},
                200:  {"accuracy": 0.767, "latency_ms": 0.27, "train_time_sec": 6},
                500:  {"accuracy": 0.858, "latency_ms": 0.27, "train_time_sec": 6},
                2000: {"accuracy": 0.897, "latency_ms": 0.27, "train_time_sec": 11},
            },
        },
        "prompt": {
            "GPT-4o-mini": {
                "zero-shot": {"accuracy": 0.844, "latency_ms": 418, "cost_per_1k": 0.016},
                "few-shot":  {"accuracy": 0.884, "latency_ms": 454, "cost_per_1k": 0.075},
            },
            "Claude-Haiku-3": {
                "zero-shot": {"accuracy": 0.810, "latency_ms": 673, "cost_per_1k": 0.033},
                "few-shot":  {"accuracy": 0.860, "latency_ms": 777, "cost_per_1k": 0.142},
            },
        },
    },
    "ner": {
        "fine_tune": {
            "Qwen2.5-1.5B": {
                50:   {"f1": 0.025, "latency_ms": 3.1,  "train_time_sec": 317},
                200:  {"f1": 0.103, "latency_ms": 3.1,  "train_time_sec": 197},
                500:  {"f1": 0.156, "latency_ms": 3.1,  "train_time_sec": 141},
                2000: {"f1": 0.384, "latency_ms": 3.1,  "train_time_sec": 140},
            },
            "Phi-3-mini": {
                50:   {"f1": 0.050, "latency_ms": 8.3,  "train_time_sec": 838},
                200:  {"f1": 0.129, "latency_ms": 8.3,  "train_time_sec": 520},
                500:  {"f1": 0.329, "latency_ms": 8.3,  "train_time_sec": 370},
                2000: {"f1": 0.474, "latency_ms": 8.3,  "train_time_sec": 364},
            },
        },
        "prompt": {
            "GPT-4o-mini": {
                "zero-shot": {"f1": 0.561, "latency_ms": 1077, "cost_per_1k": 0.038},
                "few-shot":  {"f1": 0.593, "latency_ms": 1129, "cost_per_1k": 0.054},
            },
            "Claude-Haiku-3": {
                "zero-shot": {"f1": 0.516, "latency_ms": 841,  "cost_per_1k": 0.080},
                "few-shot":  {"f1": 0.544, "latency_ms": 887,  "cost_per_1k": 0.114},
            },
        },
    },
}

TASK_ALIASES = {
    "classification":       "classification_binary",
    "clf":                  "classification_binary",
    "sentiment":            "classification_binary",
    "binary":               "classification_binary",
    "text-classification":  "classification_binary",
    "multiclass":           "classification_multiclass",
    "multi-class":          "classification_multiclass",
    "topic":                "classification_multiclass",
    "news":                 "classification_multiclass",
    "ner":                  "ner",
    "named-entity":         "ner",
    "entity":               "ner",
    "tagging":              "ner",
}

TASK_DISPLAY = {
    "classification_binary":     "Binary Classification",
    "classification_multiclass": "Multi-class Classification",
    "ner":                       "Named Entity Recognition",
}

def get_metric(task):
    return "accuracy" if "classification" in task else "f1"

def interpolate(task, model_name, n):
    data = RESULTS[task]["fine_tune"][model_name]
    keys = sorted(data.keys())
    if n <= keys[0]:
        return data[keys[0]]
    if n >= keys[-1]:
        return data[keys[-1]]
    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i+1]
        if lo <= n <= hi:
            t = (n - lo) / (hi - lo)
            metric = get_metric(task)
            return {
                metric:           round(data[lo][metric] + t * (data[hi][metric] - data[lo][metric]), 3),
                "latency_ms":     round(data[lo]["latency_ms"] + t * (data[hi]["latency_ms"] - data[lo]["latency_ms"]), 1),
                "train_time_sec": data[lo]["train_time_sec"],
            }
    return data[keys[-1]]

def build_candidates(task, n, latency_budget_ms, cost_sensitive):
    metric     = get_metric(task)
    candidates = []

    for model_name in RESULTS[task]["fine_tune"]:
        stats = interpolate(task, model_name, n)
        candidates.append({
            "approach":    "Fine-tune",
            "model":       model_name,
            "mode":        "—",
            metric:        stats[metric],
            "latency_ms":  stats["latency_ms"],
            "cost_per_1k": 0.0,
            "train_time":  stats.get("train_time_sec", "?"),
            "feasible":    stats["latency_ms"] <= latency_budget_ms,
        })

    for model_name, modes in RESULTS[task]["prompt"].items():
        for mode, stats in modes.items():
            if cost_sensitive and mode == "few-shot":
                continue
            candidates.append({
                "approach":    "Prompt",
                "model":       model_name,
                "mode":        mode,
                metric:        stats[metric],
                "latency_ms":  stats["latency_ms"],
                "cost_per_1k": stats["cost_per_1k"],
                "train_time":  "none",
                "feasible":    stats["latency_ms"] <= latency_budget_ms,
            })

    feasible = [c for c in candidates if c["feasible"]]
    pool     = feasible if feasible else candidates
    pool.sort(key=lambda x: x[metric], reverse=True)
    return pool, metric

def reasoning(task, n, best):
    if task == "ner":
        if best["approach"] == "Prompt":
            return (
                "For structured labeling tasks like NER, prompted frontier models outperform "
                "LoRA fine-tuned generative LMs at all data sizes tested (up to N=2,000). "
                "Generative models with classification heads are architecturally suboptimal for "
                "token-level sequence labeling. Consider a dedicated encoder model (e.g. BERT) "
                "if you need a local solution."
            )
        else:
            return (
                "Fine-tuning is recommended given your latency constraint, but note that NER "
                "requires significantly more data than classification tasks. "
                "Performance is poor below N=500."
            )
    elif "classification" in task:
        if best["approach"] == "Fine-tune" and best["model"] == "BERT-base":
            return (
                f"BERT-base is the optimal choice here. It trains in seconds, runs at <0.3ms/sample, "
                f"costs nothing at inference, and matches larger LoRA models at N={n}. "
                f"Don't overlook encoder-only models for classification tasks."
            )
        elif best["approach"] == "Fine-tune":
            if n < 200:
                return (
                    f"Fine-tuning is technically the best option here but N={n} is below the "
                    f"reliable threshold (~200 examples). Results will be highly variable across runs "
                    f"(observed std up to 0.08). Strongly consider prompting a frontier API instead "
                    f"until you have more data."
                )
            else:
                return (
                    f"Fine-tuned local models match or exceed frontier APIs at N={n} for classification, "
                    f"with ~50-150x lower inference latency and near-zero ongoing cost. "
                    f"One-time training cost is minimal."
                )
        else:
            if task == "classification_multiclass":
                return (
                    f"For multi-class classification with N={n}, prompted frontier models are more "
                    f"reliable. Fine-tuning becomes competitive around N=500 for Phi-3 and N=2,000 "
                    f"for smaller models. Collect more data or use API prompting for now."
                )
            else:
                return (
                    f"With N={n} training examples, prompting a frontier API is more reliable "
                    f"than fine-tuning for binary classification. The crossover point is around "
                    f"N=500 where fine-tuned models begin to match API performance."
                )
    return ""

def run(task, n, latency_budget_ms, cost_sensitive):
    candidates, metric = build_candidates(task, n, latency_budget_ms, cost_sensitive)
    best = candidates[0]

    cost_display  = f"${best['cost_per_1k']:.4f}" if best["cost_per_1k"] > 0 else "~$0 (local)"
    train_display = f"{best['train_time']}s" if isinstance(best["train_time"], (int, float)) else best["train_time"]
    color         = "green" if best["approach"] == "Fine-tune" else "magenta"
    met_val       = best[metric]

    console.print()
    console.print(Panel.fit(
        "[bold cyan]finetune-or-prompt[/bold cyan]  ·  empirical decision tool",
        border_style="cyan"
    ))

    console.print(f"\n[bold]Inputs:[/bold]")
    console.print(f"  Task:            [yellow]{TASK_DISPLAY[task]}[/yellow]")
    console.print(f"  Training data:   [yellow]{n:,} examples[/yellow]")
    console.print(f"  Latency budget:  [yellow]{latency_budget_ms}ms / sample[/yellow]")
    console.print(f"  Cost sensitive:  [yellow]{'yes' if cost_sensitive else 'no'}[/yellow]")

    console.print(f"\n[bold]Recommendation:[/bold]")
    console.print(Panel(
        f"[bold {color}]{best['approach']}[/bold {color}]  ·  "
        f"[bold white]{best['model']}[/bold white]"
        + (f"  ({best['mode']})" if best["mode"] != "—" else "")
        + f"\n\n"
        f"  {metric.upper():<12} {met_val:.3f}\n"
        f"  Latency      {best['latency_ms']:.1f}ms / sample\n"
        f"  Cost/1k inf  {cost_display}\n"
        f"  Train time   {train_display}",
        border_style=color,
        title="✅ Best option",
    ))

    console.print(f"\n[bold]All options (sorted by {metric}):[/bold]")
    table = Table(box=box.ROUNDED, header_style="bold blue")
    table.add_column("Approach",     style="cyan",    min_width=10)
    table.add_column("Model",        style="white",   min_width=16)
    table.add_column("Mode",         style="dim",     min_width=10)
    table.add_column(metric.upper(), style="green",   min_width=8,  justify="right")
    table.add_column("Latency",      style="yellow",  min_width=10, justify="right")
    table.add_column("Cost/1k inf",  style="magenta", min_width=10, justify="right")
    table.add_column("Train time",   style="dim",     min_width=10, justify="right")

    for i, c in enumerate(candidates):
        cost_str  = f"${c['cost_per_1k']:.4f}" if c["cost_per_1k"] > 0 else "~$0"
        train_str = f"{c['train_time']}s" if isinstance(c["train_time"], (int, float)) else c["train_time"]
        table.add_row(
            c["approach"], c["model"], c["mode"],
            f"{c[metric]:.3f}", f"{c['latency_ms']:.1f}ms",
            cost_str, train_str,
            style="bold" if i == 0 else "",
        )

    console.print(table)

    reason = reasoning(task, n, best)
    if reason:
        console.print(f"\n[bold]Why:[/bold]")
        console.print(f"  {reason}")

    console.print()

def main():
    parser = argparse.ArgumentParser(
        description="Should you fine-tune or prompt? An empirical recommendation tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Task options:
  classification / sentiment / binary   →  Binary classification (e.g. SST-2)
  multiclass / topic / news             →  Multi-class classification (e.g. AG News)
  ner / named-entity / tagging          →  Named entity recognition

Examples:
  python recommend.py --task classification --data 300
  python recommend.py --task multiclass --data 500 --latency 100
  python recommend.py --task ner --data 2000 --latency 500
  python recommend.py --task sentiment --data 50 --cost-sensitive
        """
    )
    parser.add_argument("--task",           required=True,            help="Task type (see options below)")
    parser.add_argument("--data",           required=True, type=int,  help="Number of training examples you have")
    parser.add_argument("--latency",        type=float, default=2000, help="Max latency per sample in ms (default: 2000)")
    parser.add_argument("--cost-sensitive", action="store_true",      help="Exclude higher-cost few-shot options")

    args = parser.parse_args()
    task = TASK_ALIASES.get(args.task.lower())

    if not task:
        console.print(f"[red]Unknown task '{args.task}'.[/red]")
        console.print(f"Valid options: {', '.join(TASK_ALIASES.keys())}")
        return

    run(
        task=task,
        n=args.data,
        latency_budget_ms=args.latency,
        cost_sensitive=args.cost_sensitive,
    )

if __name__ == "__main__":
    main()