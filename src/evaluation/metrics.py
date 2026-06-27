# src/evaluation/metrics.py
"""
Standard QA evaluation metrics (EM, F1, BLEU, ROUGE)
FIXED: Implements SQuAD-style evaluation with proper normalization
"""

import re
import string
from collections import Counter
from typing import List, Dict, Tuple
import numpy as np

def normalize_answer(s: str) -> str:
    """
    Normalize answer string for comparison (SQuAD-style)
    
    Steps:
    1. Lowercase
    2. Remove punctuation
    3. Remove articles (a, an, the)
    4. Remove extra whitespace
    
    Args:
        s: Raw answer string
        
    Returns:
        Normalized string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> bool:
    """
    Exact match metric after normalization
    
    Handles both yes/no questions and span-based answers.
    For yes/no questions, checks if prediction contains yes/no token.
    
    Args:
        prediction: Model prediction
        ground_truth: Gold standard answer
        
    Returns:
        True if normalized strings match exactly
    """
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)
    
    # Direct match
    if norm_pred == norm_gold:
        return True
    
    # Special handling for yes/no questions
    if norm_gold in ['yes', 'no']:
        # Extract yes/no from prediction
        pred_lower = norm_pred.lower()
        if norm_gold == 'yes' and 'yes' in pred_lower:
            return True
        elif norm_gold == 'no' and 'no' in pred_lower:
            return True
    
    return False


def f1_score(prediction: str, ground_truth: str) -> float:
    """
    Token-level F1 score (SQuAD metric)
    
    Computes F1 based on token overlap between prediction and ground truth.
    
    Args:
        prediction: Model prediction
        ground_truth: Gold standard answer
        
    Returns:
        F1 score between 0.0 and 1.0
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    
    # Handle empty cases
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return float(prediction_tokens == ground_truth_tokens)
    
    # Token overlap using Counter for proper handling of duplicates
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    
    return f1


def compute_metrics(predictions: List[str], 
                   ground_truths: List[str]) -> Dict[str, float]:
    """
    Compute aggregate metrics over a dataset
    
    Args:
        predictions: List of model predictions
        ground_truths: List of gold answers
        
    Returns:
        Dictionary with aggregate statistics and per-sample scores
    """
    assert len(predictions) == len(ground_truths), \
        f"Length mismatch: {len(predictions)} predictions vs {len(ground_truths)} golds"
    
    em_scores = []
    f1_scores = []
    
    for pred, gold in zip(predictions, ground_truths):
        em = exact_match(pred, gold)
        f1 = f1_score(pred, gold)
        
        em_scores.append(float(em))
        f1_scores.append(f1)
    
    return {
        'EM': np.mean(em_scores),
        'F1': np.mean(f1_scores),
        'EM_std': np.std(em_scores),
        'F1_std': np.std(f1_scores),
        'EM_scores': em_scores,
        'F1_scores': f1_scores,
        'num_samples': len(predictions)
    }


def compute_metrics_by_type(predictions: List[str],
                            ground_truths: List[str],
                            question_types: List[str]) -> Dict[str, Dict]:
    """
    Compute metrics stratified by question type
    
    Args:
        predictions: Model predictions
        ground_truths: Gold answers
        question_types: Question type labels (e.g., 'bridge', 'comparison')
        
    Returns:
        Nested dict with metrics per type
    """
    # Group by type
    by_type = {}
    for pred, gold, qtype in zip(predictions, ground_truths, question_types):
        if qtype not in by_type:
            by_type[qtype] = {'predictions': [], 'golds': []}
        by_type[qtype]['predictions'].append(pred)
        by_type[qtype]['golds'].append(gold)
    
    # Compute metrics per type
    results = {}
    for qtype, data in by_type.items():
        results[qtype] = compute_metrics(data['predictions'], data['golds'])
    
    return results


def compute_rouge_scores(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE-L scores (longest common subsequence)
    
    Args:
        predictions: Model predictions
        references: Reference answers
        
    Returns:
        Dictionary with ROUGE-L metrics
    """
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    except ImportError:
        # Fallback to simple implementation
        def simple_rouge_l(pred: str, ref: str) -> float:
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            if not ref_tokens:
                return 0.0
            lcs_len = _lcs_length(pred_tokens, ref_tokens)
            recall = lcs_len / len(ref_tokens)
            precision = lcs_len / len(pred_tokens) if pred_tokens else 0.0
            if recall + precision == 0:
                return 0.0
            return 2 * (recall * precision) / (recall + precision)
        
        scores = [simple_rouge_l(p, r) for p, r in zip(predictions, references)]
        return {
            'ROUGE-L': np.mean(scores),
            'ROUGE-L_std': np.std(scores)
        }
    
    # Using rouge_score library
    scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)['rougeL'].fmeasure
        scores.append(score)
    
    return {
        'ROUGE-L': np.mean(scores),
        'ROUGE-L_std': np.std(scores),
        'ROUGE-L_scores': scores
    }


def _lcs_length(seq1: List[str], seq2: List[str]) -> int:
    """Compute longest common subsequence length"""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i-1] == seq2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]


def compute_hallucination_metrics(
    predictions: List[str],
    contexts: List[List[str]],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute hallucination detection metrics
    
    Metrics:
    - Context overlap ratio: What fraction of prediction tokens appear in context
    - Faithfulness score: Semantic similarity between prediction and context
    - Entity-level hallucination: Proper nouns not in context (heuristic)
    
    Args:
        predictions: Model predictions
        contexts: Retrieved context passages
        references: Ground truth answers
        
    Returns:
        Dictionary with hallucination metrics
    """
    context_overlaps = []
    faithfulness_scores = []
    entity_hallucination_rates = []
    
    for pred, context_list, ref in zip(predictions, contexts, references):
        if not pred.strip():
            context_overlaps.append(0.0)
            faithfulness_scores.append(0.0)
            entity_hallucination_rates.append(0.0)
            continue
        
        # 1. Token-level context overlap
        pred_tokens = set(normalize_answer(pred).split())
        context_text = " ".join(context_list).lower()
        context_tokens = set(context_text.split())
        
        if pred_tokens:
            overlap = len(pred_tokens & context_tokens) / len(pred_tokens)
            context_overlaps.append(overlap)
        else:
            context_overlaps.append(0.0)
        
        # 2. Faithfulness via semantic similarity (if sentence-transformers available)
        try:
            from sentence_transformers import SentenceTransformer, util
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            context_text_full = " ".join(context_list)[:1000]
            pred_emb = model.encode(pred, convert_to_tensor=True)
            context_emb = model.encode(context_text_full, convert_to_tensor=True)
            
            sim = float(util.cos_sim(pred_emb, context_emb)[0][0])
            faithfulness_scores.append(sim)
        except:
            # Fallback to token overlap
            faithfulness_scores.append(overlap)
        
        # 3. Entity hallucination (simple heuristic)
        # Check if capitalized words in prediction are in context
        pred_words = pred.split()
        capitalized_in_pred = [w for w in pred_words if w and w[0].isupper()]
        
        if capitalized_in_pred:
            context_lower = context_text.lower()
            halluc_count = sum(
                1 for word in capitalized_in_pred 
                if word.lower() not in context_lower
            )
            entity_hallucination_rates.append(halluc_count / len(capitalized_in_pred))
        else:
            entity_hallucination_rates.append(0.0)
    
    return {
        'avg_context_overlap': float(np.mean(context_overlaps)) if context_overlaps else 0.0,
        'std_context_overlap': float(np.std(context_overlaps)) if context_overlaps else 0.0,
        'avg_faithfulness': float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
        'std_faithfulness': float(np.std(faithfulness_scores)) if faithfulness_scores else 0.0,
        'avg_entity_hallucination_rate': float(np.mean(entity_hallucination_rates)) if entity_hallucination_rates else 0.0,
        'low_overlap_rate': float(sum(1 for o in context_overlaps if o < 0.3) / len(context_overlaps)) if context_overlaps else 0.0,
        'low_faithfulness_rate': float(sum(1 for f in faithfulness_scores if f < 0.5) / len(faithfulness_scores)) if faithfulness_scores else 0.0
    }


def compute_semantic_entropy(predictions: List[str]) -> Dict[str, float]:
    """
    Compute semantic entropy for uncertainty estimation
    
    Groups predictions by semantic similarity and computes entropy.
    Higher entropy = more uncertainty in predictions.
    
    Args:
        predictions: List of predictions
        
    Returns:
        Dictionary with semantic entropy metrics
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = [model.encode(p) for p in predictions]
        
        # Compute pairwise similarities
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        
        # Group predictions by similarity (simple clustering)
        clusters = {}
        clustered = set()
        
        for i in range(len(predictions)):
            if i in clustered:
                continue
            
            cluster_id = len(clusters)
            cluster = [i]
            clustered.add(i)
            
            for j in range(i+1, len(predictions)):
                if j not in clustered and sim_matrix[i][j] > 0.8:
                    cluster.append(j)
                    clustered.add(j)
            
            clusters[cluster_id] = cluster
        
        # Compute entropy over cluster sizes
        cluster_sizes = np.array([len(c) for c in clusters.values()])
        cluster_probs = cluster_sizes / len(predictions)
        entropy = -np.sum(cluster_probs * np.log(cluster_probs + 1e-10))
        
        return {
            'semantic_entropy': float(entropy),
            'num_semantic_clusters': len(clusters),
            'cluster_sizes': cluster_sizes.tolist()
        }
    except:
        # Fallback: use token-level diversity
        all_tokens = set()
        for pred in predictions:
            all_tokens.update(normalize_answer(pred).split())
        
        diversity = len(all_tokens) / (sum(len(normalize_answer(p).split()) for p in predictions) + 1e-10)
        
        return {
            'semantic_entropy': float(diversity),
            'num_semantic_clusters': 1,
            'cluster_sizes': [len(predictions)]
        }


def compute_refusal_metrics(predictions: List[str]) -> Dict[str, float]:
    """
    Compute refusal and abstention metrics
    
    Args:
        predictions: List of predictions
        
    Returns:
        Dictionary with refusal metrics
    """
    refusal_phrases = [
        "i don't know",
        "cannot answer",
        "not sure",
        "unclear",
        "insufficient",
        "no information",
        "don't have",
        "cannot determine"
    ]
    
    non_empty = []
    refusals = []
    
    for pred in predictions:
        pred_lower = pred.lower().strip()
        
        if len(pred_lower) == 0:
            continue
        
        non_empty.append(pred)
        
        is_refusal = any(phrase in pred_lower for phrase in refusal_phrases)
        refusals.append(is_refusal)
    
    if non_empty:
        refusal_rate = sum(refusals) / len(refusals)
        answer_rate = len(non_empty) / len(predictions)
    else:
        refusal_rate = 0.0
        answer_rate = 0.0
    
    return {
        'refusal_rate': float(refusal_rate),
        'answer_rate': float(answer_rate),
        'num_refusals': sum(refusals),
        'num_answered': len(non_empty)
    }


def compute_length_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute length-related metrics
    
    Args:
        predictions: Model predictions
        references: Reference answers
        
    Returns:
        Dictionary with length metrics
    """
    pred_lengths = [len(p.split()) for p in predictions]
    ref_lengths = [len(r.split()) for r in references]
    
    return {
        'avg_pred_length': float(np.mean(pred_lengths)),
        'avg_ref_length': float(np.mean(ref_lengths)),
        'median_pred_length': float(np.median(pred_lengths)),
        'median_ref_length': float(np.median(ref_lengths)),
        'length_ratio': float(np.mean(pred_lengths) / (np.mean(ref_lengths) + 1e-10)),
        'overly_long_rate': float(sum(1 for p, r in zip(pred_lengths, ref_lengths) if p > 2*r) / len(predictions))
    }
