# experiments/enhanced_analysis.py
"""
Enhanced post-evaluation analysis for publication-ready results
Generates comprehensive statistics, tables, and insights
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedAnalyzer:
    """Comprehensive analysis for publication-quality results"""
    
    def __init__(self, results_dir: str = 'results/main_evaluation'):
        self.results_dir = results_dir
        self.df = None
        self.summary = None
        
    def load_results(self):
        """Load results from CSV"""
        csv_path = os.path.join(self.results_dir, 'results_500.csv')
        if os.path.exists(csv_path):
            self.df = pd.read_csv(csv_path)
            logger.info(f"Loaded {len(self.df)} results from {csv_path}")
            return True
        return False
    
    def load_summary(self):
        """Load summary JSON"""
        summary_path = os.path.join(self.results_dir, 'summary.json')
        if os.path.exists(summary_path):
            with open(summary_path, 'r') as f:
                self.summary = json.load(f)
            return True
        return False
    
    def generate_main_results_table(self) -> pd.DataFrame:
        """Generate Table 1: Main Results Comparison"""
        if self.df is None:
            return None
        
        systems = self.df['system'].unique()
        results = []
        
        for system in systems:
            sys_data = self.df[self.df['system'] == system]
            
            em = sys_data['exact_match'].mean()
            f1 = sys_data['f1_score'].mean()
            em_std = sys_data['exact_match'].std()
            f1_std = sys_data['f1_score'].std()
            
            # Hallucination proxy: overly long answers
            pred_lens = sys_data['prediction'].str.split().str.len()
            gold_lens = sys_data['gold_answer'].str.split().str.len()
            halluc_rate = (pred_lens > gold_lens * 2).mean()
            
            avg_time = sys_data['time_seconds'].mean()
            num_errors = (sys_data['exact_match'] == 0).sum()
            
            results.append({
                'System': system,
                'EM': f"{em:.1%}",
                'EM±std': f"±{em_std:.3f}",
                'F1': f"{f1:.3f}",
                'F1±std': f"±{f1_std:.3f}",
                'Hallucination Rate': f"{halluc_rate:.1%}",
                'Avg Time (s)': f"{avg_time:.2f}",
                'Failures': int(num_errors)
            })
        
        return pd.DataFrame(results)
    
    def generate_statistical_comparison(self) -> Dict:
        """Generate statistical significance tests"""
        if self.df is None:
            return None
        
        systems = self.df['system'].unique()
        comparisons = []
        
        # Get all pairwise comparisons
        for i, sys1 in enumerate(systems):
            for sys2 in systems[i+1:]:
                data1 = self.df[self.df['system'] == sys1]['exact_match'].values
                data2 = self.df[self.df['system'] == sys2]['exact_match'].values
                
                # Paired t-test
                if len(data1) == len(data2):
                    t_stat, p_value = stats.ttest_rel(data1, data2)
                    
                    # Cohen's d effect size
                    mean_diff = np.mean(data2) - np.mean(data1)
                    pooled_std = np.sqrt((np.std(data1)**2 + np.std(data2)**2) / 2)
                    cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                    
                    # Wilcoxon signed-rank test
                    w_stat, w_p = stats.wilcoxon(data1, data2)
                    
                    comparisons.append({
                        'System 1': sys1,
                        'System 2': sys2,
                        'Mean Diff (%)': f"{mean_diff*100:+.2f}%",
                        't-statistic': f"{t_stat:.3f}",
                        'p-value (t-test)': f"{p_value:.4f}",
                        'Significant (p<0.05)': 'YES' if p_value < 0.05 else 'NO',
                        'Cohen\'s d': f"{cohens_d:.3f}",
                        'Effect Size': self._interpret_cohens_d(cohens_d),
                        'Wilcoxon p-value': f"{w_p:.4f}"
                    })
        
        return comparisons
    
    def generate_per_type_analysis(self) -> Dict:
        """Generate Table 3: Per-Type Performance"""
        if self.df is None:
            return None
        
        analysis = {}
        systems = self.df['system'].unique()
        types = self.df['question_type'].unique()
        
        for qtype in types:
            type_data = self.df[self.df['question_type'] == qtype]
            type_results = []
            
            for system in systems:
                sys_type_data = type_data[type_data['system'] == system]
                if len(sys_type_data) > 0:
                    em = sys_type_data['exact_match'].mean()
                    f1 = sys_type_data['f1_score'].mean()
                    type_results.append({
                        'System': system,
                        'EM': f"{em:.1%}",
                        'F1': f"{f1:.3f}",
                        'Count': len(sys_type_data)
                    })
            
            analysis[qtype] = type_results
        
        return analysis
    
    def generate_error_analysis(self) -> Dict:
        """Generate error categorization and insights"""
        if self.df is None:
            return None
        
        analysis = {}
        systems = self.df['system'].unique()
        
        for system in systems:
            sys_data = self.df[self.df['system'] == system]
            
            # Categorize errors
            errors_by_type = {
                'Exact Failures': (sys_data['exact_match'] == 0).sum(),
                'Low F1 (<0.3)': ((sys_data['f1_score'] > 0) & (sys_data['f1_score'] < 0.3)).sum(),
                'Medium F1 (0.3-0.6)': ((sys_data['f1_score'] >= 0.3) & (sys_data['f1_score'] < 0.6)).sum(),
                'High F1 (>0.6)': (sys_data['f1_score'] >= 0.6).sum(),
            }
            
            # Length-based analysis
            pred_lens = sys_data['prediction'].str.split().str.len()
            gold_lens = sys_data['gold_answer'].str.split().str.len()
            length_diff = pred_lens - gold_lens
            
            analysis[system] = {
                'Error Categories': errors_by_type,
                'Length Analysis': {
                    'Mean Pred Length': f"{pred_lens.mean():.1f}",
                    'Mean Gold Length': f"{gold_lens.mean():.1f}",
                    'Mean Difference': f"{length_diff.mean():.1f}",
                    'Over-generation Rate (2x+)': f"{(length_diff > gold_lens).mean():.1%}"
                }
            }
        
        return analysis
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Small"
        elif abs_d < 0.5:
            return "Medium"
        elif abs_d < 0.8:
            return "Large"
        else:
            return "Very Large"
    
    def save_analysis_report(self, output_path: str = None):
        """Save comprehensive analysis report"""
        if output_path is None:
            output_path = os.path.join(self.results_dir, 'analysis_report.json')
        
        report = {
            'main_results': self.generate_main_results_table().to_dict('records') if self.df is not None else None,
            'statistical_comparison': self.generate_statistical_comparison(),
            'per_type_analysis': self.generate_per_type_analysis(),
            'error_analysis': self.generate_error_analysis(),
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Saved analysis report to {output_path}")
        return report
    
    def print_all_analyses(self):
        """Print all analyses to console"""
        print("\n" + "="*80)
        print("TABLE 1: MAIN RESULTS")
        print("="*80)
        main_table = self.generate_main_results_table()
        if main_table is not None:
            print(main_table.to_string(index=False))
        
        print("\n" + "="*80)
        print("TABLE 2: STATISTICAL SIGNIFICANCE TESTS")
        print("="*80)
        stats_comps = self.generate_statistical_comparison()
        if stats_comps:
            stats_df = pd.DataFrame(stats_comps)
            print(stats_df.to_string(index=False))
        
        print("\n" + "="*80)
        print("TABLE 3: PER-TYPE PERFORMANCE")
        print("="*80)
        per_type = self.generate_per_type_analysis()
        if per_type:
            for qtype, results in per_type.items():
                print(f"\n{qtype.upper()}:")
                type_df = pd.DataFrame(results)
                print(type_df.to_string(index=False))
        
        print("\n" + "="*80)
        print("ERROR ANALYSIS")
        print("="*80)
        error_analysis = self.generate_error_analysis()
        if error_analysis:
            error_df = pd.DataFrame([
                {
                    'System': system,
                    **analysis['Error Categories'],
                    **{f"Length-{k}": v for k, v in analysis['Length Analysis'].items()}
                }
                for system, analysis in error_analysis.items()
            ])
            print(error_df.to_string(index=False))


def main():
    """Run enhanced analysis on completed evaluation results"""
    analyzer = EnhancedAnalyzer('results/main_evaluation')
    
    if not analyzer.load_results():
        logger.error("Could not load results CSV")
        return
    
    analyzer.print_all_analyses()
    analyzer.save_analysis_report()


if __name__ == '__main__':
    main()
