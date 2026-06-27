# src/utils/data_utils.py
"""
Dataset loading and preprocessing utilities
Handles HotpotQA loading with caching and fallbacks
"""

import json
import os
import re
import string
from typing import List, Dict, Optional
from datasets import load_dataset
import logging

logger = logging.getLogger(__name__)

def normalize_answer(s: str) -> str:
    """
    Normalize answer for exact match evaluation
    Follows official SQuAD evaluation script
    """
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def load_hotpotqa(split: str = 'validation',
                  num_samples: Optional[int] = None,
                  cache_dir: str = 'data/',
                  random_seed: int = 42) -> List[Dict]:
    """
    Load HotpotQA dataset with caching
    
    Args:
        split: Dataset split ('train', 'validation')
        num_samples: Number of samples to load (None = all)
        cache_dir: Directory for caching
        random_seed: Random seed for sampling
        
    Returns:
        List of processed samples
    """
    # First try specific cache file
    cache_file = os.path.join(cache_dir, f'hotpotqa_{split}_{num_samples or "all"}.json')
    if os.path.exists(cache_file):
        logger.info(f"Loading from cache: {cache_file}")
        with open(cache_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if num_samples and len(data) > num_samples:
                return data[:num_samples]
            return data
    
    # Then try generic cache file
    generic_cache = os.path.join(cache_dir, 'hotpotqa_cache.json')
    if os.path.exists(generic_cache):
        logger.info(f"Loading from generic cache: {generic_cache}")
        with open(generic_cache, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            # Preprocess the raw data
            data = [preprocess_hotpotqa_item(item) for item in raw_data]
            if num_samples and len(data) > num_samples:
                return data[:num_samples]
            return data
    
    # Try sample cache as fallback
    sample_cache = os.path.join(cache_dir, 'hotpotqa_sample.json')
    if os.path.exists(sample_cache):
        logger.info(f"Loading from sample cache: {sample_cache}")
        with open(sample_cache, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            # Preprocess the raw data
            data = [preprocess_hotpotqa_item(item) for item in raw_data]
            if num_samples and len(data) > num_samples:
                return data[:num_samples]
            return data
    
    # Load from HuggingFace (with timeout and fallback)
    logger.info(f"Loading HotpotQA from HuggingFace (split={split})...")
    
    try:
        dataset = load_dataset('hotpot_qa', 'distractor', split=split, timeout=30)
    except Exception as e:
        logger.error(f"Failed to load from HuggingFace: {e}")
        # Create minimal test data for development
        logger.warning("Creating minimal test dataset for development...")
        data = [
            {
                "question": "Test question 1",
                "answer": "Test answer 1",
                "type": "bridge",
                "level": "medium",
                "context": {"sentences": ["Test sentence 1", "Test sentence 2"]},
            }
            for _ in range(num_samples or 5)
        ]
        return data
    
    # Sample if requested
    if num_samples is not None:
        dataset = dataset.shuffle(seed=random_seed).select(range(min(num_samples, len(dataset))))
    
    # Process samples
    samples = []
    for item in dataset:
        processed = preprocess_hotpotqa_item(item)
        samples.append(processed)
    
    logger.info(f"Loaded {len(samples)} samples from HotpotQA")
    
    # Cache for future use
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)
    logger.info(f"Cached to: {cache_file}")
    
    return samples


def preprocess_hotpotqa_item(item: Dict) -> Dict:
    """
    Preprocess a single HotpotQA item
    
    Handles different context formats and extracts relevant fields.
    
    Args:
        item: Raw HotpotQA item
        
    Returns:
        Processed item with standardized format
    """
    # Extract context sentences
    if 'context' in item:
        if isinstance(item['context'], dict):
            # Format: {'title': [...], 'sentences': [...]}
            context = item['context'].get('sentences', [])
        elif isinstance(item['context'], list):
            # Format: [['title', ['sent1', 'sent2']], ...]
            context = []
            for ctx_item in item['context']:
                if isinstance(ctx_item, list) and len(ctx_item) >= 2:
                    if isinstance(ctx_item[1], list):
                        context.extend(ctx_item[1])
                    else:
                        context.append(str(ctx_item[1]))
                elif isinstance(ctx_item, str):
                    context.append(ctx_item)
        else:
            context = [str(item['context'])]
    else:
        context = []
    
    return {
        'question': item.get('question', ''),
        'answer': item.get('answer', ''),
        'type': item.get('type', 'unknown'),
        'level': item.get('level', 'unknown'),
        'context': {'sentences': context}
    }


def stratified_sample(samples: List[Dict],
                     n: int,
                     stratify_by: str = 'type',
                     random_seed: int = 42) -> List[Dict]:
    """
    Stratified sampling to maintain type distribution
    
    Args:
        samples: List of samples
        n: Number of samples to select
        stratify_by: Field to stratify by
        random_seed: Random seed
        
    Returns:
        Stratified sample
    """
    import random
    random.seed(random_seed)
    
    # Group by type
    by_type = {}
    for sample in samples:
        key = sample.get(stratify_by, 'unknown')
        if key not in by_type:
            by_type[key] = []
        by_type[key].append(sample)
    
    # Calculate samples per type
    total = len(samples)
    result = []
    
    for type_key, type_samples in by_type.items():
        proportion = len(type_samples) / total
        n_type = int(n * proportion)
        
        # Sample from this type
        sampled = random.sample(type_samples, min(n_type, len(type_samples)))
        result.extend(sampled)
    
    # If we don't have enough, sample more from largest group
    if len(result) < n:
        remaining = n - len(result)
        largest_group = max(by_type.values(), key=len)
        additional = random.sample(largest_group, min(remaining, len(largest_group)))
        result.extend(additional)
    
    random.shuffle(result)
    return result[:n]