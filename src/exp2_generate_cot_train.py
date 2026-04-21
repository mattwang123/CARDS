wexists() and not args.test:
                try:
                    with open(ds_save_path, 'r') as f:
                        gen_results = json.load(f)
                    start_idx = len(gen_results)
                    if start_idx > 0 and start_idx < len(data):
                        print(f"    Resuming {ds_name} from query {start_idx}/{len(data)}")
                except json.JSONDecodeError:
                    print(f"    Warning: {ds_save_path} is corrupted. Starting from scratch.")

            if start_idx >= len(data):
                print(f"    - {ds_name} already completed. Skipping.")
                continue

            max_tokens = 1024 
            sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
            
            for i in tqdm(range(start_idx, len(data), save_interval), desc=f"Generating CoT for {ds_name}"):
                chunk = data[i:i + save_interval]
                prompts = [format_cot_prompt(item['question'], model_name) for item in chunk]
                
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
                
                for j, (item, prompt_text, out) in enumerate(zip(chunk, prompts, outputs)):
                    actual_idx = item.get('idx', item.get('id', i + j))
                    gen_results.append({
                        "question_idx": actual_idx, 
                        "question": item['question'],
                        "is_sufficient": item.get('is_sufficient', None),
                        "prompt": prompt_text,
                        "generated_response": out.outputs[0].text
                    })
                
                with open(ds_save_path, 'w') as f:
                    json.dump(gen_results, f, indent=2)

        # Free GPU memory heavily before loading the next model
        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='math')
    parser.add_argument('--tp', type=int, default=2) # Tensor Parallel size
    parser.add_argument('--output_dir', type=str, default='experiments/dynamic_tracking_train')
    parser.add_argument('--test', action='store_true', help="Run a quick 5-sample debug test")
    parser.add_argument('--test_samples', type=int, default=5)
    args = parser.parse_args()

    run_cot_generation(args)