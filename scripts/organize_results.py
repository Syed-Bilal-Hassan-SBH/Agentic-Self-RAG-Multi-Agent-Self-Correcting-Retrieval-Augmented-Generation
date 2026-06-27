#!/usr/bin/env python3
# scripts/organize_results.py
"""
Post-processing script to organize and visualize batch evaluation results
Creates tables, charts, and summaries from batch evaluation output
"""

import pandas as pd
import json
import os
from pathlib import Path
from tabulate import tabulate
from datetime import datetime

def organize_batch_results(results_dir='results/batch_evaluation'):
    """Organize and display batch evaluation results"""
    
    print("\n" + "="*100)
    print("BATCH EVALUATION RESULTS ORGANIZER")
    print("="*100 + "\n")
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Load summary
    summary_file = results_path / 'performance_summary_500.json'
    if summary_file.exists():
        with open(summary_file) as f:
            summary = json.load(f)
        print(f"📊 Timestamp: {summary.get('timestamp', 'N/A')}")
        print(f"📊 Total Samples: {summary.get('total_samples', 'N/A')}\n")
    
    # Load detailed results
    csv_file = results_path / 'detailed_results_500.csv'
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        
        # 1. Overall metrics
        print("="*100)
        print("1. OVERALL PERFORMANCE")
        print("="*100 + "\n")
        
        overall_metrics = df.groupby('system').agg({
            'exact_match': ['sum', 'count', lambda x: f"{x.sum()/len(x)*100:.1f}%"],
            'f1_score': 'mean',
            'time_seconds': ['mean', 'sum', 'max']
        }).round(3)
        
        systems_summary = []
        for system in df['system'].unique():
            system_df = df[df['system'] == system]
            correct = system_df['exact_match'].sum()
            total = len(system_df)
            em_pct = correct/total*100
            
            systems_summary.append({
                'System': system,
                'EM (%)': f"{em_pct:.1f}",
                'F1 Score': f"{system_df['f1_score'].mean():.3f}",
                'Avg Time (s)': f"{system_df['time_seconds'].mean():.2f}",
                'Total Time (s)': f"{system_df['time_seconds'].sum():.1f}",
                'Correct': f"{correct}/{total}"
            })
        
        print(tabulate(systems_summary, headers='keys', tablefmt='grid'))
        print()
        
        # 2. By question type
        print("="*100)
        print("2. PERFORMANCE BY QUESTION TYPE")
        print("="*100 + "\n")
        
        for qtype in df['question_type'].unique():
            type_df = df[df['question_type'] == qtype]
            print(f"\n{qtype.upper()} ({len(type_df)} samples):\n")
            
            type_summary = []
            for system in df['system'].unique():
                system_type_df = type_df[type_df['system'] == system]
                if len(system_type_df) > 0:
                    correct = system_type_df['exact_match'].sum()
                    total = len(system_type_df)
                    em_pct = correct/total*100
                    
                    type_summary.append({
                        'System': system,
                        'EM (%)': f"{em_pct:.1f}",
                        'F1': f"{system_type_df['f1_score'].mean():.3f}",
                        'Time (s)': f"{system_type_df['time_seconds'].mean():.2f}",
                        'Correct': f"{correct}/{total}"
                    })
            
            print(tabulate(type_summary, headers='keys', tablefmt='simple'))
        
        # 3. Speed comparison
        print("\n" + "="*100)
        print("3. SPEED ANALYSIS")
        print("="*100 + "\n")
        
        speed_summary = []
        baseline_time = df[df['system'] == df['system'].unique()[0]]['time_seconds'].mean()
        
        for system in sorted(df['system'].unique(), key=lambda x: df[df['system']==x]['time_seconds'].mean()):
            system_df = df[df['system'] == system]
            avg_time = system_df['time_seconds'].mean()
            speedup = baseline_time / avg_time if avg_time > 0 else 0
            
            speed_summary.append({
                'System': system,
                'Avg Time (s)': f"{avg_time:.3f}",
                'Speedup': f"{speedup:.1f}x",
                'Rank': "⚡" * int(speedup)
            })
        
        print(tabulate(speed_summary, headers='keys', tablefmt='simple'))
        
        # 4. Error analysis
        print("\n" + "="*100)
        print("4. ERROR ANALYSIS")
        print("="*100 + "\n")
        
        error_summary = []
        for system in df['system'].unique():
            system_df = df[df['system'] == system]
            incorrect = system_df[system_df['exact_match'] == False]
            
            error_rate = len(incorrect) / len(system_df) * 100
            
            error_summary.append({
                'System': system,
                'Errors': len(incorrect),
                'Total': len(system_df),
                'Error Rate (%)': f"{error_rate:.1f}"
            })
        
        print(tabulate(error_summary, headers='keys', tablefmt='simple'))
        
        # 5. Sample variance
        print("\n" + "="*100)
        print("5. PERFORMANCE VARIANCE")
        print("="*100 + "\n")
        
        variance_summary = []
        for system in df['system'].unique():
            system_df = df[df['system'] == system]
            
            variance_summary.append({
                'System': system,
                'F1 Std Dev': f"{system_df['f1_score'].std():.3f}",
                'Time Std Dev': f"{system_df['time_seconds'].std():.3f}",
                'Time Min (s)': f"{system_df['time_seconds'].min():.3f}",
                'Time Max (s)': f"{system_df['time_seconds'].max():.3f}"
            })
        
        print(tabulate(variance_summary, headers='keys', tablefmt='simple'))
        
        # 6. Top performers
        print("\n" + "="*100)
        print("6. TOP PERFORMING SAMPLES")
        print("="*100 + "\n")
        
        best_samples = df[df['exact_match'] == True].nlargest(5, 'f1_score')
        print("\nFastest Correct Answers:")
        for idx, row in df[(df['exact_match'] == True)].nsmallest(5, 'time_seconds').iterrows():
            print(f"  • {row['system']:<25} Time: {row['time_seconds']:.3f}s | Q: {row['question'][:60]}")
        
        # 7. Problem cases
        print("\n" + "="*100)
        print("7. PROBLEM CASES")
        print("="*100 + "\n")
        
        worst_samples = df[df['exact_match'] == False].nlargest(5, 'time_seconds')
        print("\nSlowest Incorrect Answers:")
        for idx, row in worst_samples.iterrows():
            print(f"  • {row['system']:<25} Time: {row['time_seconds']:.3f}s")
            print(f"    Q: {row['question'][:70]}")
            print(f"    Expected: {row['gold_answer']} | Got: {row['prediction'][:50]}\n")
    
    # Load side-by-side comparison
    sidebyside_file = results_path / 'sidebyside_comparison_500.txt'
    if sidebyside_file.exists():
        print("\n" + "="*100)
        print("8. DETAILED SIDE-BY-SIDE COMPARISON")
        print("="*100)
        with open(sidebyside_file) as f:
            print(f.read()[:2000] + "\n... (see sidebyside_comparison_500.txt for full details)")
    
    print("\n" + "="*100)
    print("✅ RESULTS ORGANIZED")
    print("="*100)
    print(f"\nFull results available in: {results_dir}/")
    print("\nKey files:")
    print("  • detailed_results_500.csv - Machine-readable data")
    print("  • sidebyside_comparison_500.txt - Human-readable comparison")
    print("  • system_comparison_500.txt - Breakdown by question type")
    print("  • detailed_log_500.txt - Complete sample logs")
    print("  • performance_summary_500.json - Structured metrics\n")

if __name__ == '__main__':
    organize_batch_results()
