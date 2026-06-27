#!/usr/bin/env python3
# scripts/download_hotpotqa.py
"""
Download and cache HotpotQA dataset
Fallback script if HuggingFace datasets fails
"""

import json
import os
import sys
from datasets import load_dataset
from tqdm import tqdm

def download_hotpotqa(output_dir: str = 'data/'):
    """Download HotpotQA and save to local cache"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("📥 Downloading HotpotQA dataset from HuggingFace...")
    
    try:
        # Load validation split
        dataset = load_dataset('hotpot_qa', 'distractor', split='validation')
        
        print(f"✓ Loaded {len(dataset)} validation samples")
        
        # Convert to list of dicts
        samples = []
        for item in tqdm(dataset, desc="Processing"):
            # Extract context
            if isinstance(item['context'], dict):
                context = item['context'].get('sentences', [])
            elif isinstance(item['context'], list):
                context = []
                for ctx in item['context']:
                    if isinstance(ctx, list) and len(ctx) >= 2:
                        if isinstance(ctx[1], list):
                            context.extend(ctx[1])
            else:
                context = []
            
            samples.append({
                'question': item['question'],
                'answer': item['answer'],
                'type': item['type'],
                'level': item.get('level', 'unknown'),
                'context': {'sentences': context}
            })
        
        # Save to cache
        cache_file = os.path.join(output_dir, 'hotpotqa_cache.json')
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(samples, f, indent=2)
        
        print(f"✓ Saved to: {cache_file}")
        print(f"✓ Total samples: {len(samples)}")
        
        # Print statistics
        type_counts = {}
        for s in samples:
            t = s['type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        print("\n📊 Dataset statistics:")
        for t, count in type_counts.items():
            print(f"  {t}: {count} ({count/len(samples)*100:.1f}%)")
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n💡 Alternative: Download manually from:")
        print("   https://hotpotqa.github.io/")
        sys.exit(1)


if __name__ == "__main__":
    download_hotpotqa()