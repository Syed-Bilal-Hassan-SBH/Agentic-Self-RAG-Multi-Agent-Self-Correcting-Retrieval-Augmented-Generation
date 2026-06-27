#!/usr/bin/env python3
"""
Debug script to identify and fix EM metric bug
The issue: EM=0% but F1>0% indicates string comparison problem
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluation.metrics import normalize_answer, exact_match, f1_score
import json

# Load actual results from smoke test
with open('results/main_evaluation/results_500.csv', 'r') as f:
    import csv
    reader = csv.DictReader(f)
    rows = list(reader)

print("=" * 80)
print("METRIC DEBUG REPORT")
print("=" * 80)

# Test 1: Check normalize_answer function
print("\n1. TESTING normalize_answer() FUNCTION")
print("-" * 80)

test_cases = [
    "The answer is Paris",
    "paris",
    "Paris, France",
    "the quick brown fox",
    "123 Main Street",
    "Dr. Smith",
    "It's a test",
]

for test in test_cases:
    normalized = normalize_answer(test)
    print(f"Input:      '{test}'")
    print(f"Normalized: '{normalized}'")
    print()

# Test 2: Check actual predictions vs gold answers
print("\n2. ANALYZING ACTUAL PREDICTIONS")
print("-" * 80)

sample_count = min(10, len(rows))
perfect_matches = []
em_failures = []

for i, row in enumerate(rows[:sample_count]):
    pred = row['prediction']
    gold = row['gold_answer']
    
    em = exact_match(pred, gold)
    f1 = float(row['f1_score'])
    
    print(f"\nSample {i}:")
    print(f"  Prediction: '{pred[:60]}...'")
    print(f"  Gold:       '{gold[:60]}...'")
    print(f"  EM Score:   {em} | F1 Score: {f1:.4f}")
    
    # Detailed normalization check
    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)
    
    print(f"  Norm Pred:  '{norm_pred[:60]}...'")
    print(f"  Norm Gold:  '{norm_gold[:60]}...'")
    print(f"  Match?:     {norm_pred == norm_gold}")
    
    if not em and f1 > 0:
        em_failures.append({
            'index': i,
            'pred': pred,
            'gold': gold,
            'f1': f1,
            'pred_norm': norm_pred,
            'gold_norm': norm_gold
        })
    elif em:
        perfect_matches.append(i)

print(f"\n\n3. SUMMARY STATISTICS")
print("-" * 80)
print(f"Total samples analyzed: {len(rows)}")
print(f"Perfect EM matches: {len(perfect_matches)}")
print(f"EM=0 but F1>0 cases: {len(em_failures)}")

# Test 3: Check for common normalization issues
print(f"\n\n4. CHECKING FOR COMMON NORMALIZATION ISSUES")
print("-" * 80)

issues = {
    'empty_predictions': 0,
    'only_punctuation': 0,
    'only_articles': 0,
    'case_sensitive_match': 0,
    'article_handling': 0,
}

for row in rows:
    pred = row['prediction'].strip()
    gold = row['gold_answer'].strip()
    
    if not pred:
        issues['empty_predictions'] += 1
    
    norm_pred = normalize_answer(pred)
    norm_gold = normalize_answer(gold)
    
    if not norm_pred:
        issues['only_punctuation'] += 1
    
    if norm_pred.lower() == norm_gold.lower():
        if norm_pred != norm_gold:
            issues['case_sensitive_match'] += 1

for issue, count in issues.items():
    if count > 0:
        print(f"  {issue}: {count} instances")

# Test 4: Compare with simple string matching
print(f"\n\n5. ALTERNATIVE MATCHING STRATEGIES")
print("-" * 80)

def simple_match(pred, gold):
    """Simplest possible matching"""
    return pred.strip().lower() == gold.strip().lower()

def token_match(pred, gold):
    """Token-based matching"""
    pred_tokens = set(pred.lower().split())
    gold_tokens = set(gold.lower().split())
    if not gold_tokens:
        return len(pred_tokens) == 0
    return pred_tokens == gold_tokens

simple_em = sum(1 for row in rows if simple_match(row['prediction'], row['gold_answer']))
token_em = sum(1 for row in rows if token_match(row['prediction'], row['gold_answer']))
current_em = sum(1 for row in rows if exact_match(row['prediction'], row['gold_answer']))

print(f"Current EM implementation: {current_em}/{len(rows)} ({100*current_em/len(rows):.1f}%)")
print(f"Simple case-insensitive:   {simple_em}/{len(rows)} ({100*simple_em/len(rows):.1f}%)")
print(f"Token-based matching:      {token_em}/{len(rows)} ({100*token_em/len(rows):.1f}%)")

# Test 5: Print recommendations
print(f"\n\n6. RECOMMENDATIONS")
print("-" * 80)

if current_em == 0 and simple_em > 0:
    print("✗ ISSUE IDENTIFIED: normalize_answer() is too aggressive")
    print("  - Simple case-insensitive matching has some matches")
    print("  - Problem likely in punctuation or article removal")
    print("  - SOLUTION: Review and simplify normalize_answer()")
    
elif current_em == 0 and simple_em == 0:
    print("✗ ISSUE IDENTIFIED: Predictions are fundamentally wrong")
    print("  - Even simple matching produces no matches")
    print("  - Problem is in the prediction generation itself")
    print("  - SOLUTION: Debug the answer generation pipeline")
    
elif current_em == 0 and token_em > 0:
    print("✗ ISSUE IDENTIFIED: Token-level matching works but exact fails")
    print("  - Exact word matching is too strict")
    print("  - Problem likely in punctuation handling")
    print("  - SOLUTION: Relax exact matching or fix punctuation removal")

print("\n" + "=" * 80)
