"""Statistical comparison between ANN and SNN performance"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy import stats as scipy_stats


def paired_ttest(ann_scores: List[float], snn_scores: List[float]) -> Dict[str, Any]:
    """
    Perform paired t-test between ANN and SNN scores.

    H0: Mean difference = 0 (no difference)
    H1: Mean difference ≠ 0 (two-sided)

    Returns:
        Dictionary with t-statistic, p-value, and interpretation
    """
    if len(ann_scores) != len(snn_scores):
        raise ValueError("Scores must have same length for paired test")

    ann_arr = np.array(ann_scores)
    snn_arr = np.array(snn_scores)
    differences = ann_arr - snn_arr

    # Perform paired t-test
    t_stat, p_value = scipy_stats.ttest_rel(ann_arr, snn_arr)

    # Effect size (Cohen's d for paired samples)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    cohen_d = mean_diff / std_diff if std_diff > 0 else 0

    # Interpretation
    alpha = 0.05
    significant = p_value < alpha

    if significant:
        if np.mean(snn_arr) > np.mean(ann_arr):
            direction = "SNN better"
        else:
            direction = "ANN better"
    else:
        direction = "No significant difference"

    return {
        'test': 'paired_ttest',
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'degrees_of_freedom': len(ann_scores) - 1,
        'mean_difference': float(mean_diff),
        'std_difference': float(std_diff),
        'cohen_d': float(cohen_d),
        'significant': significant,
        'direction': direction,
        'alpha': alpha
    }


def wilcoxon_test(ann_scores: List[float], snn_scores: List[float]) -> Dict[str, Any]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    More robust to non-normal distributions.
    """
    if len(ann_scores) != len(snn_scores):
        raise ValueError("Scores must have same length")

    stat, p_value = scipy_stats.wilcoxon(ann_scores, snn_scores)

    return {
        'test': 'wilcoxon_signed_rank',
        'statistic': float(stat),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
    }


def mcnemar_test(ann_correct: List[bool], snn_correct: List[bool]) -> Dict[str, Any]:
    """
    Perform McNemar's test for comparing two classifiers on the same samples.

    Args:
        ann_correct: Boolean list of whether ANN was correct on each sample
        snn_correct: Boolean list of whether SNN was correct on each sample

    Returns:
        Dictionary with test results
    """
    if len(ann_correct) != len(snn_correct):
        raise ValueError("Lists must have same length")

    # Build contingency table
    # Both correct | Ann correct only
    # Snn correct only | Both incorrect
    both_correct = sum(a and s for a, s in zip(ann_correct, snn_correct))
    ann_only = sum(a and not s for a, s in zip(ann_correct, snn_correct))
    snn_only = sum(not a and s for a, s in zip(ann_correct, snn_correct))
    both_incorrect = sum(not a and not s for a, s in zip(ann_correct, snn_correct))

    # McNemar test statistic (with continuity correction)
    n = ann_only + snn_only
    if n == 0:
        return {
            'test': 'mcnemar',
            'statistic': 0.0,
            'p_value': 1.0,
            'significant': False,
            'contingency_table': {
                'both_correct': both_correct,
                'ann_only': ann_only,
                'snn_only': snn_only,
                'both_incorrect': both_incorrect
            }
        }

    # Chi-square with continuity correction
    chi2 = (abs(ann_only - snn_only) - 1) ** 2 / n
    p_value = 1 - scipy_stats.chi2.cdf(chi2, df=1)

    return {
        'test': 'mcnemar',
        'statistic': float(chi2),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'contingency_table': {
            'both_correct': both_correct,
            'ann_only': ann_only,
            'snn_only': snn_only,
            'both_incorrect': both_incorrect
        }
    }


def bootstrap_confidence_interval(
    ann_scores: List[float],
    snn_scores: List[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Compute bootstrap confidence intervals for the difference in means.

    Returns:
        Dictionary with CI and point estimates
    """
    ann_arr = np.array(ann_scores)
    snn_arr = np.array(snn_scores)
    n = len(ann_arr)

    # Bootstrap resampling
    diffs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        ann_sample = ann_arr[indices]
        snn_sample = snn_arr[indices]
        diffs.append(np.mean(ann_sample) - np.mean(snn_sample))

    diffs = np.array(diffs)
    lower = np.percentile(diffs, 100 * alpha / 2)
    upper = np.percentile(diffs, 100 * (1 - alpha / 2))

    mean_diff = np.mean(ann_arr) - np.mean(snn_arr)

    return {
        'method': 'bootstrap',
        'n_bootstrap': n_bootstrap,
        'mean_difference': float(mean_diff),
        'ci_lower': float(lower),
        'ci_upper': float(upper),
        'ci_width': float(upper - lower),
        'includes_zero': lower <= 0 <= upper,
        'confidence_level': 1 - alpha
    }


def compute_all_statistics(
    ann_scores: List[float],
    snn_scores: List[float],
    ann_correct: Optional[List[bool]] = None,
    snn_correct: Optional[List[bool]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive statistical comparison.

    Args:
        ann_scores: Performance scores for ANN (lower is better for loss, higher for success)
        snn_scores: Performance scores for SNN
        ann_correct: Boolean correctness indicators (optional, for McNemar)
        snn_correct: Boolean correctness indicators (optional, for McNemar)

    Returns:
        Dictionary with all statistical test results
    """
    results = {
        'ann_mean': float(np.mean(ann_scores)),
        'ann_std': float(np.std(ann_scores)),
        'snn_mean': float(np.mean(snn_scores)),
        'snn_std': float(np.std(snn_scores)),
    }

    # Parametric test (paired t-test)
    try:
        results['paired_ttest'] = paired_ttest(ann_scores, snn_scores)
    except Exception as e:
        results['paired_ttest'] = {'error': str(e)}

    # Non-parametric test (Wilcoxon)
    try:
        results['wilcoxon'] = wilcoxon_test(ann_scores, snn_scores)
    except Exception as e:
        results['wilcoxon'] = {'error': str(e)}

    # Bootstrap CI
    try:
        results['bootstrap_ci'] = bootstrap_confidence_interval(ann_scores, snn_scores)
    except Exception as e:
        results['bootstrap_ci'] = {'error': str(e)}

    # McNemar's test if correctness data provided
    if ann_correct is not None and snn_correct is not None:
        try:
            results['mcnemar'] = mcnemar_test(ann_correct, snn_correct)
        except Exception as e:
            results['mcnemar'] = {'error': str(e)}

    # Overall summary
    all_tests = [k for k in results.keys() if isinstance(results[k], dict) and 'significant' in results[k]]
    significant_count = sum(1 for k in all_tests if results[k].get('significant', False))
    total_tests = len(all_tests)

    results['summary'] = {
        'significant_tests': significant_count,
        'total_tests': total_tests,
        'overall_significant': significant_count > 0,
        'recommendation': "SNN shows significant improvement" if significant_count > 0 and results.get('bootstrap_ci', {}).get('includes_zero', True) == False and results['bootstrap_ci']['mean_difference'] < 0 else "No clear winner"
    }

    return results


def print_statistics_summary(results: Dict[str, Any]):
    """Pretty print statistical comparison results"""
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON: ANN vs SNN")
    print("="*70)

    print(f"\nPerformance Means:")
    print(f"  ANN: {results['ann_mean']:.4f} ± {results['ann_std']:.4f}")
    print(f"  SNN: {results['snn_mean']:.4f} ± {results['snn_std']:.4f}")
    print(f"  Difference: {results['ann_mean'] - results['snn_mean']:.4f}")

    if 'paired_ttest' in results and 'error' not in results['paired_ttest']:
        pt = results['paired_ttest']
        print(f"\nPaired t-test:")
        print(f"  t({pt['degrees_of_freedom']}) = {pt['t_statistic']:.3f}, p = {pt['p_value']:.4f}")
        print(f"  Cohen's d = {pt['cohen_d']:.3f} ({_interpret_effect_size(pt['cohen_d'])})")
        print(f"  {pt['direction']} (α=0.05)")

    if 'bootstrap_ci' in results and 'error' not in results['bootstrap_ci']:
        ci = results['bootstrap_ci']
        print(f"\nBootstrap 95% CI ({ci['n_bootstrap']} samples):")
        print(f"  Difference: {ci['mean_difference']:.4f}")
        print(f"  95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        print(f"  Includes zero: {ci['includes_zero']}")

    if 'summary' in results:
        print(f"\nConclusion: {results['summary']['recommendation']}")

    print("="*70 + "\n")


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size"""
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"