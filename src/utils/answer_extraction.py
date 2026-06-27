#!/usr/bin/env python3
"""
ANSWER EXTRACTION MODULE - Improved concise answer extraction
This module provides better prompting and post-processing for extracting
concise answers suitable for exact match evaluation.
"""

def create_answer_extraction_prompt():
    """
    Improved prompt for extracting concise answers
    Returns a prompt template that forces concise, direct answers
    """
    return """Use the following context to answer the question directly and concisely.

IMPORTANT:
1. Provide ONLY the answer itself, nothing more
2. Do NOT explain or provide reasoning
3. If multiple valid answers exist, provide the most specific one
4. If the answer requires a full phrase, keep it as short as possible
5. If you cannot find the answer in the context, respond with exactly: "I don't know"

Context: {context}

Question: {question}

Answer (one or two words maximum):"""


def create_verification_prompt():
    """Prompt for answer verification with confidence scoring"""
    return """You are an expert fact-checker. Given a question, retrieved context, and a proposed answer, 
determine if the answer is supported by the context and provide a confidence score.

Question: {question}
Proposed Answer: {answer}
Context: {context}

Respond in JSON format:
{{
    "is_supported": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}

Response:"""


def extract_answer_string(response_text: str, max_tokens: int = 10) -> str:
    """
    Extract just the answer from potentially verbose LLM output
    IMPROVED: Better entity extraction and answer identification
    
    Args:
        response_text: Full LLM response
        max_tokens: Maximum number of tokens in valid answer
        
    Returns:
        Extracted answer string (cleaned and normalized for comparison)
    """
    # Clean input
    response_text = response_text.strip()
    if not response_text:
        return "I don't know"
    
    # Split by common delimiters
    lines = response_text.split('\n')
    
    # Get first non-empty line as primary candidate
    answer = ""
    for line in lines:
        line = line.strip()
        if line and not line.startswith('#') and not line.startswith('!'):
            answer = line
            break
    
    if not answer:
        answer = response_text
    
    # Remove common prefixes that are part of explanatory text
    prefixes = [
        "Answer:",
        "A:",
        "The answer is:",
        "According to the context:",
        "Based on the context:",
        "The answer is",
        "Response:",
        "Final answer:",
    ]
    
    for prefix in prefixes:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
            break
    
    # If answer still contains explanation markers, extract just the main entity
    # E.g., "Shirley Temple held the government positions of Chief of Protocol" -> "Chief of Protocol"
    if any(marker in answer.lower() for marker in [' is ', ' are ', ' was ', ' were ', 'is the', 'are the', 'the ']):
        # Try to find noun phrases (likely to be the answer)
        # Look for capitalized sequences as potential entities
        import re
        
        # Find all capitalized sequences (potential proper nouns)
        capitalized = re.findall(r'(?:^|[\s,])[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*', answer)
        if capitalized:
            # Use the longest capitalized sequence
            entity = max(capitalized, key=len).strip()
            if entity and len(entity.split()) <= max_tokens:
                answer = entity
    
    # Remove common explanatory suffixes
    suffixes = [
        ", who held the position",
        ", which held",
        ", as referenced",
        " according to the context",
        " in the context provided",
    ]
    
    for suffix in suffixes:
        if answer.lower().endswith(suffix.lower()):
            answer = answer[:-len(suffix)].strip()
    
    # Truncate if too long
    tokens = answer.split()
    if len(tokens) > max_tokens:
        # Try to keep complete noun phrase
        # Keep first max_tokens words
        answer = ' '.join(tokens[:max_tokens])
    
    # Remove trailing punctuation only if explanatory
    if answer.endswith('.'):
        # Check if it looks like end of explanation
        if any(word in answer.lower() for word in ['such as', 'including', 'like', 'example']):
            answer = answer[:-1]
    
    # Clean up spacing
    answer = ' '.join(answer.split())
    
    # Handle special cases
    if not answer or answer.lower() in ['i don\'t know', 'unknown', 'not found']:
        return "I don't know"
    
    return answer.strip()


if __name__ == "__main__":
    # Test the extraction logic
    test_cases = [
        "Answer: Paris",
        "The answer is Paris, France",
        "Yes, they were both American directors",
        "I don't know",
        "According to the context, the answer is Paris",
        "The answer is something complicated that needs explanation",
    ]
    
    print("Testing answer extraction:")
    print("=" * 80)
    for test in test_cases:
        extracted = extract_answer_string(test)
        print(f"Input:  {test}")
        print(f"Output: {extracted}")
        print()
