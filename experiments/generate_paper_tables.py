# experiments/generate_paper_tables.py
"""
Generate publication-ready tables and figures from evaluation results
Produces LaTeX tables, markdown tables, and plots for paper figures
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from src.utils.logging_utils import setup_logger

logger = setup_logger("paper_tables", log_file="logs/paper_tables.log")


def load_evaluation_results(result_dir: str = "results/main_evaluation") -> Dict:
    """Load evaluation results from JSON files"""
    logger.info(f"Loading results from {result_dir}...")
    
    summary_path = os.path.join(result_dir, 'summary.json')
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"✓ Loaded results for {len(results)} systems")
    return results


def load_baseline_results(result_dir: str = "results/baselines") -> Dict:
    """Load baseline comparison results"""
    logger.info(f"Loading baseline results from {result_dir}...")
    
    summary_path = os.path.join(result_dir, 'baseline_comparison_summary.json')
    if not os.path.exists(summary_path):
        logger.warning(f"Baseline summary not found: {summary_path}")
        return {}
    
    with open(summary_path, 'r') as f:
        results = json.load(f)
    
    logger.info(f"✓ Loaded baseline results for {len(results)} systems")
    return results


def create_main_results_table(results: Dict) -> str:
    """Create main results table (Table 1 in paper)"""
    
    logger.info("Creating main results table...")
    
    # Extract metrics
    rows = []
    for system_name, result in results.items():
        metrics = result['overall_metrics']
        by_type = result['by_type_metrics']
        avg_time = result['avg_time']
        
        row = {
            'System': system_name,
            'EM': f"{metrics['EM']:.1%}",
            'F1': f"{metrics['F1']:.3f}",
            'EM-Std': f"±{metrics['EM_std']:.3f}",
            'F1-Std': f"±{metrics['F1_std']:.3f}",
            'Latency (s)': f"{avg_time:.2f}",
            'N': int(metrics['num_samples'])
        }
        
        # Add per-type metrics if available
        if by_type:
            for qtype in ['bridge', 'comparison']:
                if qtype in by_type:
                    type_metrics = by_type[qtype]
                    row[f'{qtype.capitalize()} EM'] = f"{type_metrics['EM']:.1%}"
                    row[f'{qtype.capitalize()} F1'] = f"{type_metrics['F1']:.3f}"
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Generate LaTeX table
    latex_table = df.to_latex(index=False, escape=False)
    
    return latex_table, df


def create_ablation_table(results: Dict) -> str:
    """Create ablation study table comparing variants"""
    
    logger.info("Creating ablation study table...")
    
    # Try to load ablation results
    ablation_dir = "results/ablations"
    if not os.path.exists(ablation_dir):
        logger.warning("Ablation results directory not found")
        return None, None
    
    ablation_summary = os.path.join(ablation_dir, 'ablation_summary.json')
    if os.path.exists(ablation_summary):
        with open(ablation_summary, 'r') as f:
            ablations = json.load(f)
        
        rows = []
        for config_name, result in ablations.items():
            metrics = result.get('metrics', {})
            row = {
                'Configuration': config_name,
                'EM': f"{metrics.get('EM', 0):.1%}",
                'F1': f"{metrics.get('F1', 0):.3f}",
                'Latency (s)': f"{metrics.get('avg_latency', 0):.2f}"
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        latex_table = df.to_latex(index=False, escape=False)
        return latex_table, df
    
    return None, None


def create_error_analysis_table(result_dir: str = "results/main_evaluation") -> str:
    """Create error analysis table showing error types and frequencies"""
    
    logger.info("Creating error analysis table...")
    
    error_file = os.path.join(result_dir, 'error_analysis.json')
    if not os.path.exists(error_file):
        logger.warning("Error analysis file not found")
        return None
    
    with open(error_file, 'r') as f:
        error_data = json.load(f)
    
    # Aggregate error statistics
    error_types = {}
    for system_name, errors in error_data.items():
        for error in errors:
            error_type = error.get('error_type', 'unknown')
            if error_type not in error_types:
                error_types[error_type] = {}
            if system_name not in error_types[error_type]:
                error_types[error_type][system_name] = 0
            error_types[error_type][system_name] += 1
    
    # Create dataframe
    df = pd.DataFrame(error_types).fillna(0).astype(int)
    latex_table = df.to_latex(escape=False)
    
    return latex_table, df


def create_statistical_comparison_table(result_dir: str = "results/main_evaluation") -> str:
    """Create statistical significance table"""
    
    logger.info("Creating statistical comparison table...")
    
    comp_file = os.path.join(result_dir, 'statistical_comparisons.json')
    if not os.path.exists(comp_file):
        logger.warning("Statistical comparisons file not found")
        return None
    
    with open(comp_file, 'r') as f:
        comparisons = json.load(f)
    
    rows = []
    for comparison in comparisons:
        row = {
            'Comparison': f"{comparison['system_a_name']} vs {comparison['system_b_name']}",
            'Δ Mean': f"{comparison['difference']:+.3f}",
            'p-value': f"{comparison['p_value_wilcoxon']:.4f}",
            "Cohen's d": f"{comparison['cohen_d']:.3f}",
            'Significant': '✓' if comparison['significant_wilcoxon'] else '✗'
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    latex_table = df.to_latex(index=False, escape=False)
    
    return latex_table, df


def plot_results_comparison(results: Dict, output_dir: str = "results/paper_figures"):
    """Create comparison plots"""
    
    logger.info("Creating comparison plots...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    systems = list(results.keys())
    em_scores = [results[s]['overall_metrics']['EM'] for s in systems]
    f1_scores = [results[s]['overall_metrics']['F1'] for s in systems]
    latencies = [results[s]['avg_time'] for s in systems]
    
    # Figure 1: EM vs F1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = sns.color_palette("husl", len(systems))
    
    # EM comparison
    ax1.barh(systems, em_scores, color=colors)
    ax1.set_xlabel('Exact Match (%)', fontsize=12)
    ax1.set_title('EM Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlim([0, 1])
    for i, v in enumerate(em_scores):
        ax1.text(v + 0.02, i, f'{v:.1%}', va='center')
    
    # F1 comparison
    ax2.barh(systems, f1_scores, color=colors)
    ax2.set_xlabel('F1 Score', fontsize=12)
    ax2.set_title('F1 Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlim([0, 1])
    for i, v in enumerate(f1_scores):
        ax2.text(v + 0.02, i, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'results_comparison.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {fig_path}")
    plt.close()
    
    # Figure 2: Latency comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(systems, latencies, color=colors)
    ax.set_xlabel('Mean Latency (seconds)', fontsize=12)
    ax.set_title('System Latency Comparison', fontsize=14, fontweight='bold')
    for i, v in enumerate(latencies):
        ax.text(v + 0.05, i, f'{v:.2f}s', va='center')
    
    plt.tight_layout()
    fig_path = os.path.join(output_dir, 'latency_comparison.pdf')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    logger.info(f"✓ Saved: {fig_path}")
    plt.close()


def generate_markdown_report(
    results: Dict,
    ablation_results: Optional[Dict] = None,
    output_file: str = "results/PAPER_RESULTS.md"
) -> str:
    """Generate comprehensive markdown report"""
    
    logger.info(f"Generating markdown report...")
    
    content = []
    content.append("# Agentic Self-RAG: Evaluation Results\n")
    content.append(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
    
    # Executive Summary
    content.append("## Executive Summary\n\n")
    best_em_system = max(results.items(), key=lambda x: x[1]['overall_metrics']['EM'])
    best_f1_system = max(results.items(), key=lambda x: x[1]['overall_metrics']['F1'])
    
    content.append(f"- **Best EM**: {best_em_system[0]} ({best_em_system[1]['overall_metrics']['EM']:.1%})\n")
    content.append(f"- **Best F1**: {best_f1_system[0]} ({best_f1_system[1]['overall_metrics']['F1']:.3f})\n")
    content.append(f"- **Number of Systems**: {len(results)}\n\n")
    
    # Main Results Table
    content.append("## Main Results\n\n")
    _, main_df = create_main_results_table(results)
    content.append(main_df.to_markdown(index=False))
    content.append("\n\n")
    
    # Ablation Results (if available)
    if ablation_results:
        content.append("## Ablation Study\n\n")
        _, ablation_df = create_ablation_table(ablation_results)
        if ablation_df is not None:
            content.append(ablation_df.to_markdown(index=False))
            content.append("\n\n")
    
    # Statistical Significance
    content.append("## Statistical Significance Testing\n\n")
    content.append("`p < 0.05` indicates statistically significant difference (Wilcoxon test)\n\n")
    stat_table, stat_df = create_statistical_comparison_table()
    if stat_df is not None:
        content.append(stat_df.to_markdown(index=False))
        content.append("\n\n")
    
    # Per-Type Analysis
    content.append("## Per-Question-Type Performance\n\n")
    any_by_type = any('by_type_metrics' in r for r in results.values())
    if any_by_type:
        for qtype in ['bridge', 'comparison']:
            type_data = []
            for sys_name, result in results.items():
                if 'by_type_metrics' in result and qtype in result['by_type_metrics']:
                    type_metrics = result['by_type_metrics'][qtype]
                    type_data.append({
                        'System': sys_name,
                        'EM': f"{type_metrics['EM']:.1%}",
                        'F1': f"{type_metrics['F1']:.3f}"
                    })
            
            if type_data:
                content.append(f"### {qtype.capitalize()} Questions\n\n")
                type_df = pd.DataFrame(type_data)
                content.append(type_df.to_markdown(index=False))
                content.append("\n\n")
    
    # Methodology
    content.append("## Methodology\n\n")
    content.append("- **Dataset**: HotpotQA (validation split)\n")
    content.append("- **Metrics**: Exact Match (EM), F1 Score\n")
    content.append("- **Statistical Test**: Wilcoxon signed-rank test\n")
    content.append("- **Significance Level**: α = 0.05\n")
    content.append("- **Effect Size**: Cohen's d\n\n")
    
    # Write to file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, 'w') as f:
        f.writelines(content)
    
    logger.info(f"✓ Saved markdown report: {output_file}")
    
    return "".join(content)


def main():
    """Generate all paper tables and figures"""
    
    logger.info(f"\n{'='*80}")
    logger.info("PAPER TABLES & FIGURES GENERATION")
    logger.info(f"{'='*80}\n")
    
    # Load results
    try:
        results = load_evaluation_results("results/main_evaluation")
    except FileNotFoundError:
        logger.error("Main evaluation results not found. Run evaluation first.")
        return
    
    # Load baseline results (optional)
    baseline_results = load_baseline_results("results/baselines")
    
    # Create output directory
    output_dir = "results/paper_tables"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate main table
    logger.info("\n1. Generating main results table...")
    main_latex, main_df = create_main_results_table(results)
    main_path = os.path.join(output_dir, 'table_main_results.tex')
    with open(main_path, 'w') as f:
        f.write(main_latex)
    logger.info(f"✓ Saved: {main_path}")
    
    # Generate statistical table
    logger.info("\n2. Generating statistical comparison table...")
    stat_latex, stat_df = create_statistical_comparison_table()
    if stat_latex:
        stat_path = os.path.join(output_dir, 'table_statistical_tests.tex')
        with open(stat_path, 'w') as f:
            f.write(stat_latex)
        logger.info(f"✓ Saved: {stat_path}")
    
    # Generate error analysis table
    logger.info("\n3. Generating error analysis table...")
    error_latex, error_df = create_error_analysis_table()
    if error_latex:
        error_path = os.path.join(output_dir, 'table_error_analysis.tex')
        with open(error_path, 'w') as f:
            f.write(error_latex)
        logger.info(f"✓ Saved: {error_path}")
    
    # Generate plots
    logger.info("\n4. Creating comparison plots...")
    plot_results_comparison(results, os.path.join(output_dir, 'figures'))
    
    # Generate markdown report
    logger.info("\n5. Generating markdown report...")
    report_content = generate_markdown_report(results, baseline_results if baseline_results else None)
    
    logger.info(f"\n{'='*80}")
    logger.info("✓ PAPER TABLE GENERATION COMPLETE")
    logger.info(f"{'='*80}")
    logger.info(f"\nOutput directory: {output_dir}/")
    logger.info(f"Files generated:")
    logger.info(f"  - table_main_results.tex")
    logger.info(f"  - table_statistical_tests.tex (if available)")
    logger.info(f"  - table_error_analysis.tex (if available)")
    logger.info(f"  - figures/results_comparison.pdf")
    logger.info(f"  - figures/latency_comparison.pdf")
    logger.info(f"  - PAPER_RESULTS.md\n")


if __name__ == "__main__":
    main()
