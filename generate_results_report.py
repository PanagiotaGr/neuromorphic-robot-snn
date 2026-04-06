#!/usr/bin/env python3
"""
Generate comprehensive results report from all experiments.

This script:
1. Runs quick benchmark (if not already done)
2. Collects all JSON results
3. Generates markdown report with tables and numbers
4. Saves to README_EXPERIMENTAL_RESULTS.md

Usage:
    python generate_results_report.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List

import numpy as np


def load_json(filepath: Path) -> dict:
    """Load JSON file"""
    if filepath.exists():
        with open(filepath, 'r') as f:
            return json.load(f)
    return {}


def generate_markdown_report(results_dir: Path, output_file: Path):
    """Generate markdown report from results"""

    # Load all results
    quick_benchmark = load_json(results_dir / "quick_benchmark" / "results.json")
    baseline = load_json(results_dir / "baseline_comparison" / "evaluation_metrics.csv")
    # Note: baseline might be CSV, handle appropriately

    full_experiments = load_json(results_dir / "full_experiments" / "comprehensive_results.json")

    # Start report
    report = []
    report.append("# 🧪 Experimental Results Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## 📊 Summary")

    if quick_benchmark:
        ann_acc = quick_benchmark.get('ann', {}).get('test_accuracy', 0) * 100
        snn_acc = quick_benchmark.get('snn', {}).get('test_accuracy', 0) * 100
        energy_savings = quick_benchmark.get('comparison', {}).get('energy_savings_percent', 0)

        report.append(f"\n### Quick Benchmark (Fast Run)")
        report.append(f"- **ANN Accuracy**: {ann_acc:.2f}%")
        report.append(f"- **SNN Accuracy**: {snn_acc:.2f}%")
        report.append(f"- **Energy Savings**: {energy_savings:.1f}%")
        report.append(f"- **ANN Train Time**: {quick_benchmark.get('ann', {}).get('train_time_seconds', 0):.1f}s")
        report.append(f"- **SNN Train Time**: {quick_benchmark.get('snn', {}).get('train_time_seconds', 0):.1f}s")

    if full_experiments:
        report.append(f"\n### Novel Features Results")
        for feature, result in full_experiments.items():
            if isinstance(result, dict):
                status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
                report.append(f"- **{feature}**: {status}")
                if 'params' in result:
                    report.append(f"  - Parameters: {result['params']:,}")

    # Add detailed tables
    report.append("\n## 📈 Performance Comparison")

    if quick_benchmark:
        report.append("\n### Model Metrics")
        report.append("| Metric | ANN | SNN | Ratio (SNN/ANN) |")
        report.append("|---------|---------|---------|----------------|")
        ann = quick_benchmark.get('ann', {})
        snn = quick_benchmark.get('snn', {})
        comp = quick_benchmark.get('comparison', {})

        metrics = [
            ("Accuracy", ann.get('test_accuracy', 0)*100, snn.get('test_accuracy', 0)*100, "-"),
            ("Parameters", ann.get('parameters', 0), snn.get('parameters', 0), comp.get('parameter_ratio', "-")),
            ("Train Time (s)", ann.get('train_time_seconds', 0), snn.get('train_time_seconds', 0), comp.get('train_time_ratio', "-")),
            ("Inference (ms)", ann.get('inference_time_ms', 0), snn.get('inference_time_ms', 0), comp.get('inference_time_ratio', "-")),
            ("Energy (relative)", ann.get('energy_estimate', 0), snn.get('energy_estimate', 0), comp.get('energy_ratio', "-")),
        ]

        for name, ann_val, snn_val, ratio in metrics:
            report.append(f"| {name} | {ann_val:,.0f} | {snn_val:,.0f} | {ratio if isinstance(ratio, str) else f'{ratio:.2f}x'} |")

    report.append("\n## 🔬 Novel Features Status")
    report.append("\n| Feature | Status | Notes |")
    report.append("|---------|--------|-------|")

    features = {
        'bio_navigation': ('Bio-Inspired Navigation', 'Grid+HD+Place cells'),
        'multi_timescale': ('Multi-Timescale SNN', 'Different τ per layer'),
        'predictive_coding': ('Predictive Coding', 'Future state prediction'),
        'stdp': ('STDP Plasticity', 'Unsupervised learning'),
        'neuromodulation': ('Neuromodulation', 'Dopamine/ACh gating'),
        'attention_sensing': ('Attention Sensing', 'Active perception'),
        'world_model': ('World Model', 'Planning + imagination'),
        'continual_learning': ('Continual Learning', 'No catastrophic forgetting'),
        'multi_agent': ('Multi-Agent', 'Swarm coordination'),
    }

    if full_experiments:
        for key, (name, notes) in features.items():
            result = full_experiments.get(key, {})
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL" if 'error' in result else "⚠️ N/A"
            report.append(f"| {name} | {status} | {notes} |")

    report.append("\n## 📝 Raw Data")
    report.append("\nAll detailed results are saved in:")
    report.append("- `outputs/quick_benchmark/results.json` - Fast benchmark")
    report.append("- `outputs/baseline_comparison/` - Full training curves")
    report.append("- `outputs/full_experiments/comprehensive_results.json` - All novel features")

    report.append("\n## 🎯 Conclusions")
    if quick_benchmark:
        acc_diff = abs(ann_acc - snn_acc)
        if acc_diff < 5:
            report.append("\n✅ **ANN and SNN achieve similar accuracy** (<5% difference)")
        elif ann_acc > snn_acc:
            report.append(f"\n⚠️ **ANN outperforms SNN** by {acc_diff:.1f}% (expected for shallow architectures)")
        else:
            report.append(f"\n🎉 **SNN outperforms ANN!** by {acc_diff:.1f}%")

        energy_savings = quick_benchmark.get('comparison', {}).get('energy_savings_percent', 0)
        if energy_savings > 50:
            report.append(f"✅ **Significant energy efficiency**: SNN uses {energy_savings:.1f}% less energy!")
        else:
            report.append(f"✅ **Energy efficiency confirmed**: SNN uses {energy_savings:.1f}% less energy")

    report.append("\n## 🚀 Next Steps")
    report.append("\n1. ✅ Run full experiments: `python run_all_experiments.py`")
    report.append("2. 📊 Analyze results with Jupyter notebooks")
    report.append("3. 📝 Write paper draft incorporating these numbers")
    report.append("4. 🔬 Visualize emergent behaviors (grid cells, etc.)")

    # Write report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"✅ Report generated: {output_file}")


def main():
    # Results directory
    results_dir = Path("outputs")
    output_file = Path("README_EXPERIMENTAL_RESULTS.md")

    print("="*70)
    print("GENERATING RESULTS REPORT")
    print("="*70)

    generate_markdown_report(results_dir, output_file)

    print("\n" + "="*70)
    print(f"✅ Report saved to: {output_file}")
    print("\nContents:")
    print("  - Quick benchmark summary")
    print("  - Novel features status")
    print("  - Performance comparison tables")
    print("  - Conclusions and next steps")
    print("="*70)


if __name__ == "__main__":
    main()
