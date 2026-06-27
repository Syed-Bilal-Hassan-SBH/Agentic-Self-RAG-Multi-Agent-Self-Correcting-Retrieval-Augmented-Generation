# src/evaluation/advanced_statistics.py
"""
Advanced statistical analysis for publication-quality results
Includes hypothesis testing, effect sizes, and comprehensive comparisons
"""

import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class AdvancedStatistics:
    """Comprehensive statistical analysis toolkit"""
    
    @staticmethod
    def paired_ttest(group1: List[float], group2: List[float], 
                    alpha: float = 0.05) -> Dict:
        """
        Perform paired t-test on two groups
        Assumes data points are paired (same samples evaluated by both systems)
        """
        if len(group1) != len(group2):
            raise ValueError("Groups must have equal length for paired test")
        
        t_stat, p_value = stats.ttest_rel(group1, group2)
        
        mean_diff = np.mean(np.array(group2) - np.array(group1))
        se_diff = np.std(np.array(group2) - np.array(group1)) / np.sqrt(len(group1))
        ci_lower = mean_diff - 1.96 * se_diff
        ci_upper = mean_diff + 1.96 * se_diff
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'mean_difference': mean_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n': len(group1)
        }
    
    @staticmethod
    def wilcoxon_test(group1: List[float], group2: List[float],
                     alpha: float = 0.05) -> Dict:
        """
        Perform Wilcoxon signed-rank test (non-parametric)
        More robust to outliers than t-test
        """
        if len(group1) != len(group2):
            raise ValueError("Groups must have equal length")
        
        w_stat, p_value = stats.wilcoxon(group1, group2)
        
        return {
            'w_statistic': w_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'n': len(group1)
        }
    
    @staticmethod
    def cohens_d(group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size for paired data
        
        Interpretation:
        - d < 0.2: Small effect
        - 0.2 <= d < 0.5: Small to medium
        - 0.5 <= d < 0.8: Medium to large
        - d >= 0.8: Large effect
        """
        g1 = np.array(group1)
        g2 = np.array(group2)
        
        # For paired data: use difference scores
        diff = g2 - g1
        
        # Standard deviation of differences
        s_diff = np.std(diff, ddof=1)
        
        # Mean of differences / std of differences
        if s_diff == 0:
            return 0.0
        
        return np.mean(diff) / s_diff
    
    @staticmethod
    def bootstrap_ci(data: List[float], n_iterations: int = 10000,
                    ci: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence interval for mean
        """
        data = np.array(data)
        bootstrap_means = []
        
        for _ in range(n_iterations):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - ci
        lower = np.percentile(bootstrap_means, alpha/2 * 100)
        upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        return lower, upper
    
    @staticmethod
    def compare_multiple_systems(system_scores: Dict[str, List[float]],
                                alpha: float = 0.05) -> pd.DataFrame:
        """
        Compare multiple systems pairwise with multiple comparison correction
        
        Args:
            system_scores: Dict mapping system names to lists of scores
            alpha: Significance level
            
        Returns:
            DataFrame with all pairwise comparisons
        """
        
        systems = list(system_scores.keys())
        n_comparisons = len(systems) * (len(systems) - 1) // 2
        
        # Bonferroni correction
        alpha_corrected = alpha / n_comparisons
        
        comparisons = []
        
        for i, sys1 in enumerate(systems):
            for sys2 in systems[i+1:]:
                scores1 = system_scores[sys1]
                scores2 = system_scores[sys2]
                
                # T-test
                t_result = AdvancedStatistics.paired_ttest(
                    scores1, scores2, alpha=alpha_corrected
                )
                
                # Wilcoxon
                w_result = AdvancedStatistics.wilcoxon_test(
                    scores1, scores2, alpha=alpha_corrected
                )
                
                # Effect size
                d = AdvancedStatistics.cohens_d(scores1, scores2)
                
                # Bootstrap CI
                mean_diff = np.mean(np.array(scores2) - np.array(scores1))
                
                comparison = {
                    'System 1': sys1,
                    'System 2': sys2,
                    'Mean Diff': mean_diff,
                    't-statistic': t_result['t_statistic'],
                    'p-value (t)': t_result['p_value'],
                    'Significant (t)': t_result['significant'],
                    'p-value (W)': w_result['p_value'],
                    'Significant (W)': w_result['significant'],
                    "Cohen's d": d,
                    'Effect Size': AdvancedStatistics._interpret_cohens_d(d),
                    'Alpha (Bonf)': alpha_corrected
                }
                
                comparisons.append(comparison)
        
        return pd.DataFrame(comparisons)
    
    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    @staticmethod
    def power_analysis(effect_size: float, n: int, alpha: float = 0.05) -> float:
        """
        Calculate statistical power for paired t-test
        """
        from scipy.stats import nct
        
        # Critical t-value
        t_crit = stats.t.ppf(1 - alpha/2, n - 1)
        
        # Non-centrality parameter
        nc_param = effect_size * np.sqrt(n)
        
        # Power = 1 - beta
        power = 1 - nct.cdf(t_crit, n - 1, nc_param) + nct.cdf(-t_crit, n - 1, nc_param)
        
        return power
    
    @staticmethod
    def minimum_sample_size(effect_size: float, desired_power: float = 0.80,
                           alpha: float = 0.05) -> int:
        """
        Calculate minimum sample size needed for desired power
        """
        from scipy.stats import nct
        
        # For paired t-test
        for n in range(5, 10000):
            t_crit = stats.t.ppf(1 - alpha/2, n - 1)
            nc_param = effect_size * np.sqrt(n)
            power = 1 - nct.cdf(t_crit, n - 1, nc_param) + nct.cdf(-t_crit, n - 1, nc_param)
            
            if power >= desired_power:
                return n
        
        return 10000


class ErrorMetricsAnalyzer:
    """Analyze error types and patterns"""
    
    @staticmethod
    def categorize_errors(predictions: List[str], gold_answers: List[str],
                         question_types: List[str]) -> Dict:
        """
        Categorize errors by type
        
        Error categories:
        - No answer: Prediction is empty or "I don't know"
        - Partial: Some correct info but incomplete
        - Wrong: Completely wrong answer
        - Hallucination: Makes up information
        """
        
        errors = {
            'no_answer': 0,
            'partial': 0,
            'wrong': 0,
            'hallucination': 0,
            'correct': 0
        }
        
        by_type = {qtype: dict(errors) for qtype in set(question_types)}
        
        for pred, gold, qtype in zip(predictions, gold_answers, question_types):
            if not pred or pred.lower() in ['i don\'t know', 'unknown', 'no']:
                category = 'no_answer'
            elif len(pred.split()) > len(gold.split()) * 2:
                category = 'hallucination'
            elif pred.lower() == gold.lower():
                category = 'correct'
            elif any(word in pred.lower() for word in gold.lower().split()):
                category = 'partial'
            else:
                category = 'wrong'
            
            errors[category] += 1
            by_type[qtype][category] += 1
        
        return {
            'overall': errors,
            'by_type': by_type
        }


class ConfidenceCalibrationAnalyzer:
    """Analyze calibration of confidence scores"""
    
    @staticmethod
    def compute_calibration_metrics(confidence_scores: List[float],
                                   correct: List[bool],
                                   n_bins: int = 10) -> Dict:
        """
        Compute calibration metrics: ECE, MCE, Brier score
        """
        
        confidence = np.array(confidence_scores)
        correct = np.array(correct)
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (confidence >= lower) & (confidence < upper)
            if np.sum(mask) > 0:
                bin_acc = np.mean(correct[mask])
                bin_conf = np.mean(confidence[mask])
                bin_weight = np.sum(mask) / len(confidence)
                ece += bin_weight * np.abs(bin_acc - bin_conf)
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
            mask = (confidence >= lower) & (confidence < upper)
            if np.sum(mask) > 0:
                bin_acc = np.mean(correct[mask])
                bin_conf = np.mean(confidence[mask])
                mce = max(mce, np.abs(bin_acc - bin_conf))
        
        # Brier Score
        brier = np.mean((confidence - correct.astype(int)) ** 2)
        
        return {
            'ECE': ece,
            'MCE': mce,
            'Brier': brier,
            'Interpretation': 'Well-calibrated' if ece < 0.1 else 'Over-confident' if ece > 0.2 else 'Reasonably calibrated'
        }


def main():
    """Example usage"""
    
    # Example data
    system_a_scores = [0, 1, 0, 1, 1, 0, 1, 1, 0, 1]  # 60% correct
    system_b_scores = [1, 1, 1, 1, 1, 0, 1, 1, 0, 1]  # 80% correct
    
    # Statistical comparison
    print("="*80)
    print("STATISTICAL ANALYSIS EXAMPLE")
    print("="*80)
    
    t_test = AdvancedStatistics.paired_ttest(system_a_scores, system_b_scores)
    print(f"\nPaired t-test:")
    print(f"  t-statistic: {t_test['t_statistic']:.3f}")
    print(f"  p-value: {t_test['p_value']:.4f}")
    print(f"  Significant: {t_test['significant']}")
    
    d = AdvancedStatistics.cohens_d(system_a_scores, system_b_scores)
    print(f"\nEffect Size (Cohen's d): {d:.3f}")
    
    # Power analysis
    power = AdvancedStatistics.power_analysis(d, len(system_a_scores))
    print(f"Statistical Power: {power:.3f}")
    
    # Sample size
    min_n = AdvancedStatistics.minimum_sample_size(0.3)
    print(f"Min samples needed for d=0.3: {min_n}")


if __name__ == '__main__':
    main()
