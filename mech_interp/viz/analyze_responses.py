"""
Analyze model responses to detect overconfidence
"""
import re
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'models'))

from answer_parser import extract_numerical_answer


def extract_ground_truth(answer_text):
    """
    Extract ground truth numerical answer from GSM8K answer format

    Args:
        answer_text: GSM8K answer with #### NUMBER format

    Returns:
        float or int: Ground truth answer
        None: If parsing fails
    """
    # Look for #### followed by number
    pattern = r'####\s*(-?\d+\.?\d*)'
    matches = re.findall(pattern, answer_text)

    if matches:
        try:
            num = float(matches[-1])
            if num.is_integer():
                return int(num)
            return num
        except ValueError:
            pass

    return None


def check_correctness(model_answer, ground_truth_answer):
    """
    Check if model answer matches ground truth

    Args:
        model_answer: Extracted numerical answer from model
        ground_truth_answer: Ground truth numerical answer

    Returns:
        bool: True if correct, False otherwise
        None: If either answer is None
    """
    if model_answer is None or ground_truth_answer is None:
        return None

    # Allow small floating point tolerance
    if isinstance(model_answer, float) or isinstance(ground_truth_answer, float):
        return abs(model_answer - ground_truth_answer) < 0.01

    return model_answer == ground_truth_answer


def analyze_responses(responses_data):
    """
    Analyze model responses to calculate metrics

    Args:
        responses_data: List of dicts with question, response, is_sufficient, etc.

    Returns:
        dict: Analysis results
    """
    # Split by sufficiency
    sufficient = [r for r in responses_data if r['is_sufficient']]
    insufficient = [r for r in responses_data if not r['is_sufficient']]

    # Analyze sufficient questions
    sufficient_results = {
        'total': len(sufficient),
        'answered': 0,
        'correct': 0,
        'incorrect': 0,
        'no_answer': 0,
        'details': []
    }

    for item in sufficient:
        model_answer = extract_numerical_answer(item['response'])
        ground_truth = extract_ground_truth(item['answer'])

        result = {
            'question': item['question'][:100] + '...',
            'model_answer': model_answer,
            'ground_truth': ground_truth,
            'response': item['response'][:200] + '...'
        }

        if model_answer is None:
            sufficient_results['no_answer'] += 1
            result['status'] = 'no_answer'
        else:
            sufficient_results['answered'] += 1
            is_correct = check_correctness(model_answer, ground_truth)

            if is_correct:
                sufficient_results['correct'] += 1
                result['status'] = 'correct'
            else:
                sufficient_results['incorrect'] += 1
                result['status'] = 'incorrect'

        sufficient_results['details'].append(result)

    # Analyze insufficient questions
    insufficient_results = {
        'total': len(insufficient),
        'answered': 0,
        'no_answer': 0,
        'overconfidence_rate': 0.0,
        'details': []
    }

    for item in insufficient:
        model_answer = extract_numerical_answer(item['response'])

        result = {
            'question': item['question'][:100] + '...',
            'original_question': item.get('original_question', '')[:100] + '...',
            'removed_value': item.get('removed_value', 'N/A'),
            'removed_description': item.get('removed_description', 'N/A'),
            'model_answer': model_answer,
            'response': item['response'][:200] + '...'
        }

        if model_answer is None:
            insufficient_results['no_answer'] += 1
            result['status'] = 'no_answer'
        else:
            insufficient_results['answered'] += 1
            result['status'] = 'answered_overconfident'

        insufficient_results['details'].append(result)

    # Calculate overconfidence rate
    if insufficient_results['total'] > 0:
        insufficient_results['overconfidence_rate'] = (
            insufficient_results['answered'] / insufficient_results['total']
        )

    # Overall summary
    summary = {
        'sufficient': sufficient_results,
        'insufficient': insufficient_results,
        'total_examples': len(responses_data)
    }

    return summary


def print_analysis_report(analysis):
    """
    Print formatted analysis report

    Args:
        analysis: Analysis results from analyze_responses
    """
    print("\n" + "="*80)
    print("MODEL RESPONSE ANALYSIS REPORT")
    print("="*80)

    # Sufficient questions
    suff = analysis['sufficient']
    print(f"\nSUFFICIENT QUESTIONS ({suff['total']} total):")
    print(f"  Answered: {suff['answered']} ({suff['answered']/suff['total']*100:.1f}%)")
    print(f"  Correct: {suff['correct']} ({suff['correct']/suff['total']*100:.1f}%)")
    print(f"  Incorrect: {suff['incorrect']} ({suff['incorrect']/suff['total']*100:.1f}%)")
    print(f"  No answer: {suff['no_answer']} ({suff['no_answer']/suff['total']*100:.1f}%)")

    if suff['answered'] > 0:
        accuracy = suff['correct'] / suff['answered'] * 100
        print(f"  Accuracy (when answered): {accuracy:.1f}%")

    # Insufficient questions
    insuff = analysis['insufficient']
    print(f"\nINSUFFICIENT QUESTIONS ({insuff['total']} total):")
    print(f"  Answered anyway: {insuff['answered']} ({insuff['answered']/insuff['total']*100:.1f}%)")
    print(f"  No answer: {insuff['no_answer']} ({insuff['no_answer']/insuff['total']*100:.1f}%)")
    print(f"  Overconfidence rate: {insuff['overconfidence_rate']*100:.1f}%")

    if insuff['overconfidence_rate'] > 0.5:
        print("\n  ⚠️  HIGH OVERCONFIDENCE DETECTED!")
        print("     Model answered >50% of insufficient questions without")
        print("     recognizing missing information.")

    print("\n" + "="*80)

    # Show some examples
    print("\nEXAMPLE SUFFICIENT RESPONSES:")
    print("-"*80)
    for i, detail in enumerate(suff['details'][:2], 1):
        print(f"\n{i}. Q: {detail['question']}")
        print(f"   GT: {detail['ground_truth']}, Model: {detail['model_answer']}, Status: {detail['status']}")

    print("\n\nEXAMPLE INSUFFICIENT RESPONSES:")
    print("-"*80)
    for i, detail in enumerate(insuff['details'][:2], 1):
        print(f"\n{i}. Q: {detail['question']}")
        print(f"   Removed: {detail['removed_value']} ({detail['removed_description']})")
        print(f"   Model answer: {detail['model_answer']}, Status: {detail['status']}")

    print("\n" + "="*80)


def save_analysis_report(analysis, output_path):
    """
    Save analysis report to text file

    Args:
        analysis: Analysis results
        output_path: Path to save report
    """
    with open(output_path, 'w') as f:
        # Sufficient
        suff = analysis['sufficient']
        f.write("="*80 + "\n")
        f.write("MODEL RESPONSE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write(f"SUFFICIENT QUESTIONS ({suff['total']} total):\n")
        f.write(f"  Answered: {suff['answered']} ({suff['answered']/suff['total']*100:.1f}%)\n")
        f.write(f"  Correct: {suff['correct']} ({suff['correct']/suff['total']*100:.1f}%)\n")
        f.write(f"  Incorrect: {suff['incorrect']} ({suff['incorrect']/suff['total']*100:.1f}%)\n")
        f.write(f"  No answer: {suff['no_answer']} ({suff['no_answer']/suff['total']*100:.1f}%)\n")

        if suff['answered'] > 0:
            accuracy = suff['correct'] / suff['answered'] * 100
            f.write(f"  Accuracy (when answered): {accuracy:.1f}%\n")

        # Insufficient
        insuff = analysis['insufficient']
        f.write(f"\nINSUFFICIENT QUESTIONS ({insuff['total']} total):\n")
        f.write(f"  Answered anyway: {insuff['answered']} ({insuff['answered']/insuff['total']*100:.1f}%)\n")
        f.write(f"  No answer: {insuff['no_answer']} ({insuff['no_answer']/insuff['total']*100:.1f}%)\n")
        f.write(f"  Overconfidence rate: {insuff['overconfidence_rate']*100:.1f}%\n")

        if insuff['overconfidence_rate'] > 0.5:
            f.write("\n⚠️  HIGH OVERCONFIDENCE DETECTED!\n")
            f.write("Model answered >50% of insufficient questions.\n")

        f.write("\n" + "="*80 + "\n")

    print(f"Report saved to: {output_path}")
