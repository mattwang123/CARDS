"""
Post-process existing binary assessment results to add comprehensive metrics
"""
import json
import os
import argparse
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix


def compute_comprehensive_metrics(results):
    """
    Compute comprehensive metrics from existing detailed results
    
    Returns:
        dict: {accuracy, f1, precision, recall, confusion_matrix}
    """
    # Extract predictions and ground truth
    predictions = []
    ground_truth_labels = []
    
    for result in results:
        # Convert to binary labels (1 = sufficient/Yes, 0 = insufficient/No)
        true_label = 1 if result['is_sufficient'] else 0
        
        # Get model prediction
        if result['judgment']['verdict'] == 'correct':
            pred_label = true_label  # Correct prediction
        elif result['judgment']['verdict'] == 'incorrect':
            pred_label = 1 - true_label  # Incorrect prediction (flip)
        else:  # error
            pred_label = 0  # Default to insufficient for errors
            
        predictions.append(pred_label)
        ground_truth_labels.append(true_label)
    
    # Compute metrics using sklearn (same as probe evaluation)
    metrics = {
        'accuracy': float(accuracy_score(ground_truth_labels, predictions)),
        'f1': float(f1_score(ground_truth_labels, predictions, average='binary', zero_division=0)),
        'precision': float(precision_score(ground_truth_labels, predictions, average='binary', zero_division=0)),
        'recall': float(recall_score(ground_truth_labels, predictions, average='binary', zero_division=0))
    }
    
    # Add confusion matrix
    cm = confusion_matrix(ground_truth_labels, predictions)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def postprocess_single_file(file_path):
    """Post-process a single result file"""
    print(f"Processing: {file_path}")
    
    # Load existing results
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Check if already processed
    if 'f1' in data:
        print(f"  ✓ Already has comprehensive metrics, skipping")
        return
    
    # Extract detailed results
    detailed_results = data.get('detailed_results', [])
    
    if not detailed_results:
        print(f"  ⚠ No detailed results found, skipping")
        return
    
    # Compute comprehensive metrics
    comprehensive_metrics = compute_comprehensive_metrics(detailed_results)
    
    # Add to existing data (preserve everything, add new metrics)
    data.update({
        # Comprehensive metrics (matching probe evaluation)
        'accuracy': comprehensive_metrics['accuracy'],
        'f1': comprehensive_metrics['f1'],
        'precision': comprehensive_metrics['precision'],
        'recall': comprehensive_metrics['recall'],
        'confusion_matrix': comprehensive_metrics['confusion_matrix'],
        
        # Rename old accuracy to avoid confusion
        'legacy_accuracy': data.get('accuracy', 0.0)
    })
    
    # Save updated file
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"  ✓ Added comprehensive metrics:")
    print(f"    Accuracy: {comprehensive_metrics['accuracy']:.4f}")
    print(f"    F1: {comprehensive_metrics['f1']:.4f}")
    print(f"    Precision: {comprehensive_metrics['precision']:.4f}")
    print(f"    Recall: {comprehensive_metrics['recall']:.4f}")


def postprocess_summary_file(summary_path):
    """Post-process the summary file"""
    print(f"\nProcessing summary: {summary_path}")
    
    # Load summary
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Check if already processed
    first_model = list(summary.keys())[0]
    first_dataset = list(summary[first_model].keys())[0]
    if 'f1' in summary[first_model][first_dataset]:
        print(f"  ✓ Summary already has comprehensive metrics")
        return
    
    # Update each model-dataset entry
    updated = False
    for model_name in summary:
        for dataset_name in summary[model_name]:
            # Load corresponding detailed file
            detailed_file = f"experiments/binary_assessment/{model_name}_{dataset_name}_binary_assessment.json"
            
            if os.path.exists(detailed_file):
                with open(detailed_file, 'r') as f:
                    detailed_data = json.load(f)
                
                # Add comprehensive metrics to summary
                if 'f1' in detailed_data:
                    summary[model_name][dataset_name].update({
                        'accuracy': detailed_data['accuracy'],
                        'f1': detailed_data['f1'],
                        'precision': detailed_data['precision'],
                        'recall': detailed_data['recall']
                    })
                    updated = True
    
    if updated:
        # Save updated summary
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"  ✓ Updated summary with comprehensive metrics")
    else:
        print(f"  ⚠ No updates needed for summary")


def main():
    parser = argparse.ArgumentParser(
        description='Post-process binary assessment results to add comprehensive metrics'
    )
    parser.add_argument('--results_dir', type=str, default='experiments/binary_assessment',
                        help='Directory containing binary assessment results')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    
    print("="*80)
    print("POST-PROCESSING BINARY ASSESSMENT RESULTS")
    print("="*80)
    print(f"Results directory: {results_dir}")
    print("="*80)
    
    # Process individual result files
    result_files = list(results_dir.glob("*_binary_assessment.json"))
    
    if not result_files:
        print("No binary assessment result files found!")
        return
    
    print(f"\nFound {len(result_files)} result files to process:")
    
    for file_path in sorted(result_files):
        postprocess_single_file(file_path)
    
    # Process summary file
    summary_file = results_dir / "binary_assessment_summary.json"
    if summary_file.exists():
        postprocess_summary_file(summary_file)
    else:
        print(f"\nSummary file not found: {summary_file}")
    
    print("\n" + "="*80)
    print("POST-PROCESSING COMPLETE!")
    print("="*80)
    print("\nUpdated files now include:")
    print("  ✓ accuracy (comprehensive)")
    print("  ✓ f1 (binary F1 score)")
    print("  ✓ precision (binary precision)")  
    print("  ✓ recall (binary recall)")
    print("  ✓ confusion_matrix")
    print("  ✓ legacy_accuracy (original accuracy)")
    print("="*80)


if __name__ == '__main__':
    main()