#!/usr/bin/env python3
"""
Generate matplotlib table visualizations as PNG images
Creates publication-ready table images from all evaluation data
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path
from datetime import datetime

# Ensure output directory exists
Path('results/table_visualizations').mkdir(parents=True, exist_ok=True)

# Color scheme for tables
COLORS = {
    'header': '#2C3E50',
    'vanilla': '#ECF0F1',
    'simplified': '#D5DBDB',
    'published': '#AED6F1',
    'agentic': '#85C1E2',
    'ultra': '#5D9FD3',
    'accent': '#3498DB',
    'highlight': '#F39C12'
}

def load_summary_data():
    """Load summary statistics from JSON"""
    with open('results/main_evaluation/summary.json', 'r') as f:
        return json.load(f)

def create_main_results_table():
    """Create main results table visualization"""
    data = load_summary_data()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    systems = list(data.keys())
    table_data = [['System', 'Exact Match', 'F1 Score', 'Avg Time (s)', 'Context Overlap', 
                   'Faithfulness', 'Entity Hallucination', 'Refusal Rate']]
    
    row_colors = ['#2C3E50']  # header
    system_colors = {
        'Vanilla RAG': '#ECF0F1',
        'Simplified Self-RAG': '#D5DBDB',
        'Published Self-RAG': '#AED6F1',
        'Agentic Self-RAG': '#85C1E2',
        'Ultra Agentic RAG': '#5D9FD3'
    }
    
    for system in systems:
        metrics = data[system]
        row = [
            system,
            f"{metrics['overall_metrics']['EM']:.3f}",
            f"{metrics['overall_metrics']['F1']:.3f}",
            f"{metrics['avg_time']:.2f}",
            f"{metrics['hallucination_metrics']['avg_context_overlap']:.3f}",
            f"{metrics['hallucination_metrics']['avg_faithfulness']:.3f}",
            f"{metrics['hallucination_metrics']['avg_entity_hallucination_rate']:.3f}",
            f"{metrics['refusal_metrics']['refusal_rate']:.3f}"
        ]
        table_data.append(row)
        row_colors.append(system_colors.get(system, '#FFFFFF'))
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.18, 0.12, 0.12, 0.12, 0.13, 0.13, 0.13, 0.12])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style cells
    for i, color in enumerate(row_colors):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor(color)
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor(color)
                cell.set_edgecolor('#2C3E50')
                cell.set_linewidth(1)
    
    plt.title('Main Evaluation Results - All Systems', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/table_visualizations/table_main_results.png', dpi=300, bbox_inches='tight')
    print('[OK] Main results table saved')
    plt.close()

def create_performance_by_type_table():
    """Create performance by question type table"""
    data = load_summary_data()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data
    systems = list(data.keys())
    table_data = [['System', 'Type', 'Exact Match', 'F1 Score', 'Samples']]
    
    row_colors = ['#2C3E50']  # header
    system_colors = {
        'Vanilla RAG': '#ECF0F1',
        'Simplified Self-RAG': '#D5DBDB',
        'Published Self-RAG': '#AED6F1',
        'Agentic Self-RAG': '#85C1E2',
        'Ultra Agentic RAG': '#5D9FD3'
    }
    
    for system in systems:
        metrics = data[system]
        for qtype in ['bridge', 'comparison']:
            by_type = metrics['by_type_metrics'][qtype]
            row = [
                system if qtype == 'bridge' else '',
                qtype.capitalize(),
                f"{by_type['EM']:.3f}",
                f"{by_type['F1']:.3f}",
                str(by_type['num_samples'])
            ]
            table_data.append(row)
            row_colors.append(system_colors.get(system, '#FFFFFF'))
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.18, 0.18, 0.14])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)
    
    # Style cells
    for i, color in enumerate(row_colors):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor(color)
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor(color)
                cell.set_edgecolor('#2C3E50')
                cell.set_linewidth(1)
    
    plt.title('Performance by Question Type', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/table_visualizations/table_by_question_type.png', dpi=300, bbox_inches='tight')
    print('[OK] Performance by question type table saved')
    plt.close()

def create_hallucination_metrics_table():
    """Create hallucination metrics table"""
    data = load_summary_data()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    systems = list(data.keys())
    table_data = [['System', 'Context Overlap', 'Faithfulness', 'Entity Hallucination', 
                   'Low Overlap Rate', 'Low Faithfulness Rate', 'Low Refusal Rate']]
    
    row_colors = ['#2C3E50']  # header
    system_colors = {
        'Vanilla RAG': '#ECF0F1',
        'Simplified Self-RAG': '#D5DBDB',
        'Published Self-RAG': '#AED6F1',
        'Agentic Self-RAG': '#85C1E2',
        'Ultra Agentic RAG': '#5D9FD3'
    }
    
    for system in systems:
        metrics = data[system]
        hall = metrics['hallucination_metrics']
        row = [
            system,
            f"{hall['avg_context_overlap']:.3f}",
            f"{hall['avg_faithfulness']:.3f}",
            f"{hall['avg_entity_hallucination_rate']:.3f}",
            f"{hall['low_overlap_rate']:.3f}",
            f"{hall['low_faithfulness_rate']:.3f}",
            f"{metrics['refusal_metrics']['refusal_rate']:.3f}"
        ]
        table_data.append(row)
        row_colors.append(system_colors.get(system, '#FFFFFF'))
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.135, 0.135, 0.16, 0.155, 0.165, 0.145])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style cells
    for i, color in enumerate(row_colors):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor(color)
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor(color)
                cell.set_edgecolor('#2C3E50')
                cell.set_linewidth(1)
    
    plt.title('Hallucination Metrics Analysis', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/table_visualizations/table_hallucination_metrics.png', dpi=300, bbox_inches='tight')
    print('[OK] Hallucination metrics table saved')
    plt.close()

def create_length_metrics_table():
    """Create length metrics table"""
    data = load_summary_data()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    systems = list(data.keys())
    table_data = [['System', 'Avg Predicted Length', 'Avg Reference Length', 
                   'Length Ratio', 'Overly Long Rate']]
    
    row_colors = ['#2C3E50']  # header
    system_colors = {
        'Vanilla RAG': '#ECF0F1',
        'Simplified Self-RAG': '#D5DBDB',
        'Published Self-RAG': '#AED6F1',
        'Agentic Self-RAG': '#85C1E2',
        'Ultra Agentic RAG': '#5D9FD3'
    }
    
    for system in systems:
        metrics = data[system]
        length = metrics['length_metrics']
        row = [
            system,
            f"{length['avg_pred_length']:.3f}",
            f"{length['avg_ref_length']:.3f}",
            f"{length['length_ratio']:.3f}",
            f"{length['overly_long_rate']:.3f}"
        ]
        table_data.append(row)
        row_colors.append(system_colors.get(system, '#FFFFFF'))
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.2, 0.2, 0.18, 0.17])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style cells
    for i, color in enumerate(row_colors):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor(color)
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor(color)
                cell.set_edgecolor('#2C3E50')
                cell.set_linewidth(1)
    
    plt.title('Answer Length Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/table_visualizations/table_length_metrics.png', dpi=300, bbox_inches='tight')
    print('[OK] Length metrics table saved')
    plt.close()

def create_refusal_metrics_table():
    """Create refusal and error metrics table"""
    data = load_summary_data()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    systems = list(data.keys())
    table_data = [['System', 'Refusal Rate', 'Answer Rate', 'Num Refusals', 
                   'Num Answered', 'Errors']]
    
    row_colors = ['#2C3E50']  # header
    system_colors = {
        'Vanilla RAG': '#ECF0F1',
        'Simplified Self-RAG': '#D5DBDB',
        'Published Self-RAG': '#AED6F1',
        'Agentic Self-RAG': '#85C1E2',
        'Ultra Agentic RAG': '#5D9FD3'
    }
    
    for system in systems:
        metrics = data[system]
        refusal = metrics['refusal_metrics']
        row = [
            system,
            f"{refusal['refusal_rate']:.3f}",
            f"{refusal['answer_rate']:.3f}",
            str(refusal['num_refusals']),
            str(refusal['num_answered']),
            str(metrics['num_errors'])
        ]
        table_data.append(row)
        row_colors.append(system_colors.get(system, '#FFFFFF'))
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.5)
    
    # Style cells
    for i, color in enumerate(row_colors):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor(color)
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor(color)
                cell.set_edgecolor('#2C3E50')
                cell.set_linewidth(1)
    
    plt.title('Refusal and Error Metrics', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/table_visualizations/table_refusal_metrics.png', dpi=300, bbox_inches='tight')
    print('[OK] Refusal metrics table saved')
    plt.close()

def create_comparison_summary():
    """Create system comparison summary"""
    with open('results/main_evaluation/statistical_comparisons.json', 'r') as f:
        comparisons = json.load(f)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [['System A', 'System B', 'Mean Diff', 'P-value (Wilcoxon)', 
                   'Cohen\'s d', 'Significant']]
    
    for comp in comparisons[:10]:  # All pairwise comparisons
        row = [
            comp['system_a_name'],
            comp['system_b_name'],
            f"{comp['difference']:+.4f}",
            f"{comp['p_value_wilcoxon']:.4f}",
            f"{comp['cohen_d']:.3f}",
            'Yes' if comp['significant_wilcoxon'] else 'No'
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.2, 0.2, 0.15, 0.18, 0.14, 0.13])
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2.2)
    
    # Style cells
    row_colors = ['#2C3E50']  # header
    for i in range(len(table_data) - 1):
        row_colors.append('#F5F5F5' if i % 2 == 0 else '#FFFFFF')
    
    for i, color in enumerate(row_colors):
        for j in range(len(table_data[0])):
            cell = table[(i, j)]
            if i == 0:
                cell.set_facecolor(color)
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor(color)
                cell.set_edgecolor('#2C3E50')
                cell.set_linewidth(0.5)
    
    plt.title('Pairwise Statistical Comparisons', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('results/table_visualizations/table_statistical_comparisons.png', dpi=300, bbox_inches='tight')
    print('[OK] Statistical comparisons table saved')
    plt.close()

def main():
    print('\n' + '='*80)
    print('GENERATING MATPLOTLIB TABLE VISUALIZATIONS')
    print('='*80 + '\n')
    
    try:
        create_main_results_table()
        create_performance_by_type_table()
        create_hallucination_metrics_table()
        create_length_metrics_table()
        create_refusal_metrics_table()
        create_comparison_summary()
        
        print('\n' + '='*80)
        print('[OK] ALL VISUALIZATIONS GENERATED SUCCESSFULLY')
        print('='*80)
        print('\nGenerated 6 high-resolution PNG tables:')
        print('  1. table_main_results.png')
        print('  2. table_by_question_type.png')
        print('  3. table_hallucination_metrics.png')
        print('  4. table_length_metrics.png')
        print('  5. table_refusal_metrics.png')
        print('  6. table_statistical_comparisons.png')
        print('\nSaved to: results/table_visualizations/')
        print('Resolution: 300 DPI (publication quality)')
        print('='*80 + '\n')
        
    except Exception as e:
        print(f'[ERROR] Failed to generate visualizations: {e}')
        raise

if __name__ == '__main__':
    main()
