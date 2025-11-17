"""
Parse numerical answers from model responses
"""
import re


def extract_answer_from_boxed(text):
    """
    Extract numerical answer from \boxed{} format

    Args:
        text: Model response text

    Returns:
        float or int: Extracted answer
        None: If no answer found
    """
    # Look for \boxed{...}
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)

    if not matches:
        return None

    # Take the last match (final answer)
    answer_str = matches[-1].strip()

    # Try to extract number from the answer string
    # Handle formats like: "42", "$42", "42 dollars", etc.
    number_pattern = r'-?\d+\.?\d*'
    numbers = re.findall(number_pattern, answer_str)

    if not numbers:
        return None

    # Take the first number found
    try:
        num = float(numbers[0])
        # Return as int if it's a whole number
        if num.is_integer():
            return int(num)
        return num
    except ValueError:
        return None


def extract_answer_from_hash(text):
    """
    Extract numerical answer from #### format (GSM8K style)

    Args:
        text: Model response text

    Returns:
        float or int: Extracted answer
        None: If no answer found
    """
    # Look for #### followed by number
    pattern = r'####\s*(-?\d+\.?\d*)'
    matches = re.findall(pattern, text)

    if not matches:
        return None

    # Take the last match
    try:
        num = float(matches[-1])
        if num.is_integer():
            return int(num)
        return num
    except ValueError:
        return None


def extract_numerical_answer(text):
    """
    Try multiple extraction methods to get numerical answer

    Args:
        text: Model response text

    Returns:
        float or int: Extracted answer
        None: If no answer found
    """
    # Try \boxed{} first
    answer = extract_answer_from_boxed(text)
    if answer is not None:
        return answer

    # Try #### format
    answer = extract_answer_from_hash(text)
    if answer is not None:
        return answer

    return None


def has_answer(text):
    """
    Check if response contains an answer

    Args:
        text: Model response text

    Returns:
        bool: True if answer found, False otherwise
    """
    return extract_numerical_answer(text) is not None

def extract_binary_answer(text):
    """
    Extract Yes/No answer from \\boxed{} format (reuses same logic as extract_numerical_answer)
    
    Args:
        text: Model response text
        
    Returns:
        str: 'Yes', 'No', or None if not found
    """
    # Reuse existing boxed extraction logic
    pattern = r'\\boxed\{([^}]+)\}'
    matches = re.findall(pattern, text)

    if matches:
        answer_str = matches[-1].strip().lower()
        if 'yes' in answer_str:
            return 'Yes'
        elif 'no' in answer_str:
            return 'No'
    
    return None


if __name__ == '__main__':
    # Test cases
    test_cases = [
        ("The answer is \\boxed{42}", 42),
        ("Therefore, \\boxed{3.14}", 3.14),
        ("Final answer: \\boxed{100 dollars}", 100),
        ("#### 500", 500),
        ("The calculation gives #### 25", 25),
        ("No answer here", None),
        ("Multiple: \\boxed{10} but actually \\boxed{20}", 20),  # Takes last
    ]

    print("Testing answer parser...")
    for text, expected in test_cases:
        result = extract_numerical_answer(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{text[:50]}...' => {result} (expected: {expected})")
