import json

def normalize_answer(answer):
    """Normalize answer to boolean for comparison"""
    if isinstance(answer, bool):
        return answer
    if isinstance(answer, str):
        answer = answer.strip().lower()
        if answer in ['yes', 'y', 'true', '1']:
            return True
        elif answer in ['no', 'n', 'false', '0']:
            return False
    return None

def calculate_accuracy(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total = len(data)
    correct = 0
    errors = []
    
    for idx, item in enumerate(data):
        is_sufficient = normalize_answer(item.get('is_sufficient'))
        model_output = normalize_answer(item.get('model_output'))
        
        # Check if both are valid
        if is_sufficient is None or model_output is None:
            errors.append(f"Line {idx}: Invalid format - is_sufficient: {item.get('is_sufficient')}, model_output: {item.get('model_output')}")
            continue
        
        # Compare
        if is_sufficient == model_output:
            correct += 1
        else:
            errors.append(f"Line {idx}: Mismatch - is_sufficient: {item.get('is_sufficient')}, model_output: {item.get('model_output')}")
    
    accuracy = correct / total if total > 0 else 0
    
    print(f"Total: {total}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.2%}")

    
    return accuracy

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        json_file = 'experiments/verbalization/math/Qwen2.5-14B/umwp_generations.json'
    
    try:
        calculate_accuracy(json_file)
    except FileNotFoundError:
        print(f"Error: File '{json_file}' not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{json_file}'")