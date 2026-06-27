# src/evaluation/statistical_tests.py
"""
Statistical significance testing for system comparison
FIXED: Complete implementation with paired tests and effect sizes
"""

from scipy import stats
import numpy as np
from typing import List, Dict, Tuple

def paired_t_test(scores_a: List[float], 
                  scores_b: List[float]) -> Tuple[float, float]:
    """
    Paired t-test for dependent samples
    
    Use when comparing two systems on the same test set.
    
    Args:
        scores_a: Scores from system A
        scores_b: Scores from system B (same samples)
        
    Returns:
        (t_statistic, p_value)
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    return float(t_stat), float(p_value)


def mann_whitney_u_test(scores_a: List[float],
                       scores_b: List[float]) -> Tuple[float, float]:
    """
    Mann-Whitney U test for independent samples (non-parametric)
    
    Use when samples are not paired or normality assumption violated.
    
    Args:
        scores_a: Scores from system A
        scores_b: Scores from system B
        
    Returns:
        (u_statistic, p_value)
    """
    u_stat, p_value = stats.mannwhitneyu(scores_a, scores_b, alternative='two-sided')
    return float(u_stat), float(p_value)


def wilcoxon_test(scores_a: List[float],
                  scores_b: List[float]) -> Tuple[float, float]:
    """
    Wilcoxon signed-rank test (non-parametric paired test)
    
    Alternative to paired t-test when normality assumption violated.
    
    Args:
        scores_a: Scores from system A
        scores_b: Scores from system B (same samples)
        
    Returns:
        (statistic, p_value)
    """
    # Handle edge case where all differences are zero
    differences = [a - b for a, b in zip(scores_a, scores_b)]
    if all(d == 0 for d in differences):
        # If all differences are zero, p-value is 1.0 (no difference)
        return 0.0, 1.0
    
    try:
        stat, p_value = stats.wilcoxon(scores_a, scores_b, zero_method='zsplit')
    except ValueError:
        # Fallback if zsplit doesn't work
        try:
            stat, p_value = stats.wilcoxon(scores_a, scores_b, zero_method='pratt')
        except ValueError:
            # Last resort: use method that excludes zeros
            stat, p_value = 0.0, 1.0
    
    return float(stat), float(p_value)


def cohen_d(scores_a: List[float], scores_b: List[float]) -> float:
    """
    Cohen's d effect size
    
    Interpretation:
    - Small: |d| = 0.2
    - Medium: |d| = 0.5
    - Large: |d| = 0.8
    
    Args:
        scores_a: Scores from system A
        scores_b: Scores from system B
        
    Returns:
        Effect size (positive means B > A)
    """
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    std_a = np.std(scores_a, ddof=1)
    std_b = np.std(scores_b, ddof=1)
    n_a = len(scores_a)
    n_b = len(scores_b)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    
    # Cohen's d
    d = (mean_b - mean_a) / pooled_std
    
    return float(d)


def compare_systems(system_a_scores: List[float],
                   system_b_scores: List[float],
                   system_a_name: str = "System A",
                   system_b_name: str = "System B",
                   alpha: float = 0.05) -> Dict:
    """
    Complete statistical comparison between two systems
    
    Performs:
    - Paired t-test
    - Wilcoxon test
    - Cohen's d effect size
    - 95% confidence interval
    
    Args:
        system_a_scores: Scores from system A
        system_b_scores: Scores from system B
        system_a_name: Name of system A
        system_b_name: Name of system B
        alpha: Significance level
        
    Returns:
        Dictionary with all statistics
    """
    # Paired t-test
    t_stat, p_value_t = paired_t_test(system_a_scores, system_b_scores)
    
    # Wilcoxon test
    w_stat, p_value_w = wilcoxon_test(system_a_scores, system_b_scores)
    
    # Effect size
    effect_size = cohen_d(system_a_scores, system_b_scores)
    
    # Summary statistics
    mean_a = np.mean(system_a_scores)
    mean_b = np.mean(system_b_scores)
    std_a = np.std(system_a_scores)
    std_b = np.std(system_b_scores)
    
    # Confidence interval for difference
    diff = mean_b - mean_a
    se_diff = np.sqrt(std_a**2/len(system_a_scores) + std_b**2/len(system_b_scores))
    ci_lower = diff - 1.96 * se_diff
    ci_upper = diff + 1.96 * se_diff
    
    result = {
        'system_a_name': system_a_name,
        'system_b_name': system_b_name,
        'mean_a': float(mean_a),
        'mean_b': float(mean_b),
        'std_a': float(std_a),
        'std_b': float(std_b),
        'difference': float(diff),
        'ci_95_lower': float(ci_lower),
        'ci_95_upper': float(ci_upper),
        't_statistic': float(t_stat),
        'p_value_t_test': float(p_value_t),
        'wilcoxon_statistic': float(w_stat),
        'p_value_wilcoxon': float(p_value_w),
        'cohen_d': float(effect_size),
        'significant_t_test': p_value_t < alpha,
        'significant_wilcoxon': p_value_w < alpha,
        'alpha': alpha
    }
    
    return result


def print_comparison(result: Dict):
    """Pretty print comparison results"""
    print(f"\n{'='*70}")
    print(f"{result['system_a_name']} vs {result['system_b_name']}")
    print(f"{'='*70}")
    print(f"  {result['system_a_name']:>20}: {result['mean_a']:.3f} ± {result['std_a']:.3f}")
    print(f"  {result['system_b_name']:>20}: {result['mean_b']:.3f} ± {result['std_b']:.3f}")
    print(f"  {'Difference':>20}: {result['difference']:+.3f}")
    print(f"  {'95% CI':>20}: [{result['ci_95_lower']:+.3f}, {result['ci_95_upper']:+.3f}]")
    print(f"  {'t-statistic':>20}: {result['t_statistic']:.3f}")
    print(f"  {'p-value (t-test)':>20}: {result['p_value_t_test']:.4f}")
    print(f"  {'p-value (Wilcoxon)':>20}: {result['p_value_wilcoxon']:.4f}")
    cohens_d_label = "Cohen's d"
    print(f"  {cohens_d_label:>20}: {result['cohen_d']:.3f}")
    
    # Interpretation
    if result['significant_t_test']:
        print(f"  {'Result (t-test)':>20}: [SIGNIFICANT] (p<{result['alpha']})")
    else:
        print(f"  {'Result (t-test)':>20}: [NOT SIGNIFICANT] (p>={result['alpha']})")
    
    # Effect size interpretation
    d = abs(result['cohen_d'])
    if d < 0.2:
        effect = "negligible"
    elif d < 0.5:
        effect = "small"
    elif d < 0.8:
        effect = "medium"
    else:
        effect = "large"
    print(f"  {'Effect size':>20}: {effect}")
    print(f"{'='*70}\n")


def bootstrap_confidence_interval(scores: List[float],
                                 n_bootstrap: int = 10000,
                                 ci: float = 0.95) -> Tuple[float, float]:
    """
    Bootstrap confidence interval for mean
    
    Args:
        scores: List of scores
        n_bootstrap: Number of bootstrap samples
        ci: Confidence interval level (0.95 for 95% CI)
        
    Returns:
        (lower_bound, upper_bound)
    """
    bootstrap_means = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_bootstrap):
        bootstrap_sample = rng.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(bootstrap_sample))
    
    bootstrap_means = np.array(bootstrap_means)
    alpha = 1 - ci
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return float(lower), float(upper)


def effect_size_hedges_g(scores_a: List[float], scores_b: List[float]) -> float:
    """
    Hedges' g effect size (less biased than Cohen's d for small samples)
    
    Args:
        scores_a: Scores from system A
        scores_b: Scores from system B
        
    Returns:
        Effect size
    """
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)
    std_a = np.std(scores_a, ddof=1)
    std_b = np.std(scores_b, ddof=1)
    n_a = len(scores_a)
    n_b = len(scores_b)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
    
    # Correction factor
    correction = 1 - (3 / (4 * (n_a + n_b - 2) - 1))
    
    # Hedges' g
    g = correction * (mean_b - mean_a) / pooled_std
    
    return float(g)


def power_analysis(effect_size: float, n: int, alpha: float = 0.05) -> float:
    """
    Post-hoc power analysis
    
    Calculates the power of a paired t-test given effect size, sample size, and alpha.
    
    Args:
        effect_size: Cohen's d
        n: Number of samples
        alpha: Significance level
        
    Returns:
        Statistical power (0 to 1)
    """
    from scipy.stats import nct
    
    # Non-centrality parameter
    lambda_param = effect_size * np.sqrt(n)
    
    # Critical t-value
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # Power = P(|T| > t_crit | H1 is true)
    power = 1 - nct.cdf(t_crit, df, lambda_param) + nct.cdf(-t_crit, df, lambda_param)
    
    return float(power)


def multi_comparison_correction(p_values: List[float], 
                               method: str = 'bonferroni') -> Dict:
    """
    Multiple comparison correction for p-values
    
    Methods:
    - bonferroni: Divide alpha by number of comparisons (most conservative)
    - holm: Step-down Bonferroni (less conservative)
    - benjamini_hochberg: False Discovery Rate (less conservative)
    
    Args:
        p_values: List of p-values
        method: Correction method
        
    Returns:
        Dictionary with corrected p-values and significant tests
    """
    from scipy.stats import beta
    
    p_values = np.array(p_values)
    n_tests = len(p_values)
    
    if method == 'bonferroni':
        corrected_pvalues = p_values * n_tests
        corrected_pvalues = np.clip(corrected_pvalues, 0, 1)
    
    elif method == 'holm':
        # Holm's step-down method
        sorted_indices = np.argsort(p_values)
        corrected_pvalues = np.zeros_like(p_values)
        for i, idx in enumerate(sorted_indices):
            corrected_pvalues[idx] = p_values[idx] * (n_tests - i)
        corrected_pvalues = np.clip(corrected_pvalues, 0, 1)
    
    elif method == 'benjamini_hochberg':
        # Benjamini-Hochberg FDR correction
        sorted_indices = np.argsort(p_values)
        corrected_pvalues = np.zeros_like(p_values)
        for i, idx in enumerate(sorted_indices):
            corrected_pvalues[idx] = p_values[idx] * n_tests / (i + 1)
        corrected_pvalues = np.clip(corrected_pvalues, 0, 1)
    
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    return {
        'method': method,
        'original_pvalues': [float(p) for p in p_values],
        'corrected_pvalues': [float(p) for p in corrected_pvalues],
        'n_tests': n_tests,
        'significant_original': [float(p) < 0.05 for p in p_values],
        'significant_corrected': [float(p) < 0.05 for p in corrected_pvalues]
    }


def normality_test(scores: List[float]) -> Dict:
    """
    Test for normality using Shapiro-Wilk test
    
    Args:
        scores: List of scores
        
    Returns:
        Dictionary with test results
    """
    stat, p_value = stats.shapiro(scores)
    
    return {
        'test': 'Shapiro-Wilk',
        'statistic': float(stat),
        'p_value': float(p_value),
        'normal': p_value > 0.05
    }


def compare_multiple_systems(results: Dict[str, List[float]], 
                            alpha: float = 0.05) -> Dict:
    """
    ANOVA-based comparison of multiple systems
    
    Args:
        results: Dictionary mapping system names to score lists
        alpha: Significance level
        
    Returns:
        Dictionary with ANOVA results and post-hoc tests
    """
    system_names = list(results.keys())
    score_lists = list(results.values())
    
    # ANOVA
    f_stat, p_value_anova = stats.f_oneway(*score_lists)
    
    # Effect size (eta-squared)
    grand_mean = np.mean(np.concatenate(score_lists))
    ss_between = sum(len(scores) * (np.mean(scores) - grand_mean)**2 for scores in score_lists)
    ss_total = sum(np.sum((scores - grand_mean)**2) for scores in score_lists)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    comparison_result = {
        'method': 'One-way ANOVA',
        'f_statistic': float(f_stat),
        'p_value': float(p_value_anova),
        'eta_squared': float(eta_squared),
        'significant': p_value_anova < alpha,
        'n_systems': len(system_names),
        'systems': system_names
    }
    
    # Pairwise comparisons
    pairwise = []
    for i, sys_a in enumerate(system_names):
        for sys_b in system_names[i+1:]:
            scores_a = results[sys_a]
            scores_b = results[sys_b]
            
            # Use Welch's t-test (doesn't assume equal variances)
            t_stat, p_value = stats.ttest_ind(scores_a, scores_b, equal_var=False)
            
            pairwise.append({
                'system_a': sys_a,
                'system_b': sys_b,
                't_statistic': float(t_stat),
                'p_value': float(p_value)
            })
    
    comparison_result['pairwise'] = pairwise
    
    return comparison_result