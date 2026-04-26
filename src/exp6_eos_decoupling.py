"""
================================================================================
EXPERIMENT 6: Latent-Text Decoupling & Unified EOS Evaluation
================================================================================
1. Evaluates the frozen Unified Probe on EOS embeddings to prove geometric stationarity.
2. Cross-tabulates EOS latent predictions with text-based quadrant outcomes to expose
   the Representation-Verbalization Gap (Tragic Lying in Q1).
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import f1_score

# --- 配置参数 ---
MODELS = [
    'Qwen/Qwen2.5-Math-1.5B', 'Qwen/Qwen2.5-Math-1.5B-Instruct',
    'Qwen/Qwen2.5-3B', 'Qwen/Qwen2.5-3B-Instruct',
    'google/gemma-3-4b-it',
    'Qwen/Qwen2.5-Math-7B', 'Qwen/Qwen2.5-Math-7B-Instruct',
    'meta-llama/Meta-Llama-3.1-8B', 'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'google/gemma-3-12b-it',
    'allenai/Olmo-3-7B-Think', 'allenai/Olmo-3-7B-Instruct',
    'deepseek-ai/deepseek-math-7b-instruct',
    'Qwen/Qwen2.5-14B', 'Qwen/Qwen2.5-14B-Instruct',
    'google/gemma-3-27b-it', 'allenai/Olmo-3-32B-Think',
    'openai/gpt-oss-20b', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B',
    'deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 'Qwen/Qwen2.5-72B-Instruct'
]

DATASETS = ['umwp']#, 'treecut'
EXPORT_BASE = '/export/fs06/hwang302/CARDS'
BASE_DIR = os.path.join(EXPORT_BASE, 'exp_temporal_new')

def run_exp6_decoupling():
    for dataset in DATASETS:
        print(f"\n{'='*50}\nProcessing Dataset: {dataset.upper()}\n{'='*50}")
        
        # 加载 Exp 5 数据以便追加 unified_f1_eos
        exp5_path = os.path.join(BASE_DIR, 'results', f'exp5_global_dynamics_{dataset}.json')
        if not os.path.exists(exp5_path):
            print(f"Skipping {dataset}, Exp 5 results not found.")
            continue
            
        with open(exp5_path, 'r') as f:
            exp5_data = json.load(f)

        decoupling_results = []

        for model_name in MODELS:
            model_slug = model_name.split('/')[-1]
            if model_name not in exp5_data:
                continue
                
            print(f"Analyzing {model_slug}...")

            # 1. 加载文本评估结果 (获取真值标签和四象限分类)
            eval_path = os.path.join(EXPORT_BASE, f"experiments/dynamic_tracking_test_evaluation/math/{model_slug}/{dataset}_evaluated_traces.json")
            if not os.path.exists(eval_path):
                continue
                
            with open(eval_path, 'r') as f:
                eval_data = json.load(f).get("data", [])
                
            quadrants = np.array([item['epistemic_quadrant'] for item in eval_data])
            true_labels = np.array([1 if not item['is_sufficient'] else 0 for item in eval_data])

            # 2. 加载 EOS 测试集特征向量
            emb_path_test = os.path.join(BASE_DIR, 'embeddings', dataset, model_slug, "t_eos_test.npy")
            if not os.path.exists(emb_path_test):
                continue
            X_eos_all = np.load(emb_path_test)

            metrics = exp5_data[model_name]
            unified_layer = metrics.get("unified_layer")
            eos_layer = metrics.get("eos_layer")

            # --- 任务 A: 计算并追加 Unified Probe 在 EOS 上的 F1 ---
            if unified_layer is not None:
                unified_probe_path = os.path.join(BASE_DIR, 'probes', dataset, model_slug, f"unified_probe_layer{unified_layer}.joblib")
                if os.path.exists(unified_probe_path):
                    unified_probe = joblib.load(unified_probe_path)
                    X_test_unified = X_eos_all[:, unified_layer, :]
                    
                    # 预测并计算 F1
                    unified_eos_preds = unified_probe.predict(X_test_unified)
                    unified_f1_eos = f1_score(true_labels, unified_eos_preds)
                    
                    # 补回 Exp 5 的 JSON
                    exp5_data[model_name]["unified_f1_eos"] = float(unified_f1_eos)

            # --- 任务 B: EOS Probe 的 Decoupling 交叉分析 ---
            if eos_layer is not None:
                eos_probe_path = os.path.join(BASE_DIR, 'probes', dataset, model_slug, f"eos_probe_layer{eos_layer}.joblib")
                if os.path.exists(eos_probe_path):
                    eos_probe = joblib.load(eos_probe_path)
                    X_test_eos = X_eos_all[:, eos_layer, :]
                    
                    # 获取隐空间预测结果 (1=Probe 认为是 Insufficient)
                    latent_preds = eos_probe.predict(X_test_eos)

                    def get_decoupling_stats(q_name):
                        q_mask = (quadrants == q_name)
                        total_n = np.sum(q_mask)
                        if total_n == 0: return 0, 0, 0.0
                        
                        # 在这个象限里，探针有多少次亮起了“条件不足”的红灯
                        probe_says_insuff = np.sum(latent_preds[q_mask] == 1)
                        pct = (probe_says_insuff / total_n) * 100
                        return int(total_n), int(probe_says_insuff), float(pct)

                    q1_tot, q1_ins, q1_pct = get_decoupling_stats('Q1_Hallucination')
                    q2_tot, q2_ins, q2_pct = get_decoupling_stats('Q2_Correct_Rejection')
                    q3_tot, q3_ins, q3_pct = get_decoupling_stats('Q3_Solved_Correctly')
                    q4_tot, q4_ins, q4_pct = get_decoupling_stats('Q4_Competence_Failure')

                    decoupling_results.append({
                        "Dataset": dataset,
                        "Model": model_slug,
                        "EOS_Global_F1": round(metrics.get("eos_f1", 0), 4),
                        "Unified_EOS_F1": round(exp5_data[model_name].get("unified_f1_eos", 0), 4),
                        "Q1(Text=Number)_Total_N": q1_tot,
                        "Q1_Latent_Says_Insuff_%": round(q1_pct, 1),
                        "Q2(Text=Reject)_Total_N": q2_tot,
                        "Q2_Latent_Says_Insuff_%": round(q2_pct, 1),
                        "Q3_Latent_Says_Insuff_%": round(q3_pct, 1),
                        "Q4_Latent_Says_Insuff_%": round(q4_pct, 1),
                    })
                    
                    # 仅打印高亮结果
                    if q1_tot > 0:
                        print(f"   -> [Decoupling] In Q1 (Text Hallucinated Answer), Latent state correctly identified {q1_pct:.1f}% as Insufficient!")

        # 1. 保存更新后的 Exp 5 JSON (追加了 unified_f1_eos)
        with open(exp5_path, 'w') as f:
            json.dump(exp5_data, f, indent=2)
            
        # 2. 保存 Exp 6 的脱节分析结果
        df = pd.DataFrame(decoupling_results)
        csv_out = os.path.join(BASE_DIR, 'results', f'exp6_decoupling_{dataset}.csv')
        df.to_csv(csv_out, index=False)
        print(f"\n[SUCCESS] Exp 5 Updated and Exp 6 Decoupling results saved to: {csv_out}")

if __name__ == '__main__':
    run_exp6_decoupling()