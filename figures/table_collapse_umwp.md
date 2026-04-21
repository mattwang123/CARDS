### Markdown Format
| Model | Initial F1 ($t=0$) | Collapse F1 ($t=64$) | $\Delta$ F1 | Optimal Layer Shift |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Llama-70B | 0.919 | 0.797 | -0.122 | L34 $\rightarrow$ L36 |
| DeepSeek-R1-Distill-Qwen-32B | 0.942 | 0.813 | -0.129 | L45 $\rightarrow$ L45 |
| Meta-Llama-3.1-8B | 0.847 | 0.638 | -0.209 | L17 $\rightarrow$ L14 |
| Meta-Llama-3.1-8B-Instruct | 0.863 | 0.713 | -0.150 | L24 $\rightarrow$ L14 |
| Olmo-3-32B-Think | 0.939 | 0.809 | -0.131 | L28 $\rightarrow$ L24 |
| Olmo-3-7B-Instruct | 0.881 | 0.735 | -0.146 | L22 $\rightarrow$ L19 |
| Olmo-3-7B-Think | 0.879 | 0.740 | -0.140 | L18 $\rightarrow$ L17 |
| Qwen2.5-14B | 0.931 | 0.749 | -0.182 | L30 $\rightarrow$ L30 |
| Qwen2.5-14B-Instruct | 0.944 | 0.781 | -0.164 | L31 $\rightarrow$ L29 |
| Qwen2.5-3B | 0.882 | 0.688 | -0.193 | L27 $\rightarrow$ L23 |
| Qwen2.5-3B-Instruct | 0.896 | 0.695 | -0.200 | L30 $\rightarrow$ L26 |
| Qwen2.5-72B-Instruct | 0.955 | 0.812 | -0.144 | L58 $\rightarrow$ L59 |
| Qwen2.5-Math-1.5B | 0.905 | 0.703 | -0.202 | L20 $\rightarrow$ L16 |
| Qwen2.5-Math-1.5B-Instruct | 0.903 | 0.717 | -0.186 | L19 $\rightarrow$ L18 |
| Qwen2.5-Math-7B | 0.913 | 0.772 | -0.141 | L19 $\rightarrow$ L19 |
| Qwen2.5-Math-7B-Instruct | 0.921 | 0.785 | -0.136 | L19 $\rightarrow$ L21 |
| deepseek-math-7b-instruct | 0.867 | 0.696 | -0.171 | L18 $\rightarrow$ L15 |
| gemma-3-12b-it | 0.906 | 0.773 | -0.133 | L28 $\rightarrow$ L25 |
| gemma-3-27b-it | 0.928 | 0.840 | -0.088 | L36 $\rightarrow$ L33 |
| gemma-3-4b-it | 0.849 | 0.692 | -0.157 | L16 $\rightarrow$ L17 |
| gpt-oss-20b | 0.891 | 0.751 | -0.140 | L21 $\rightarrow$ L16 |


### LaTeX Format
\begin{table}[h!]
\centering
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Initial F1 ($t=0$)} & \textbf{Collapse F1 ($t=64$)} & \textbf{$\Delta$ F1} & \textbf{Layer Shift} \\
\midrule
DeepSeek-R1-Distill-Llama-70B & 0.919 & 0.797 & -0.122 & L34 $\rightarrow$ L36 \\
DeepSeek-R1-Distill-Qwen-32B & 0.942 & 0.813 & -0.129 & L45 $\rightarrow$ L45 \\
Meta-Llama-3.1-8B & 0.847 & 0.638 & -0.209 & L17 $\rightarrow$ L14 \\
Meta-Llama-3.1-8B-Instruct & 0.863 & 0.713 & -0.150 & L24 $\rightarrow$ L14 \\
Olmo-3-32B-Think & 0.939 & 0.809 & -0.131 & L28 $\rightarrow$ L24 \\
Olmo-3-7B-Instruct & 0.881 & 0.735 & -0.146 & L22 $\rightarrow$ L19 \\
Olmo-3-7B-Think & 0.879 & 0.740 & -0.140 & L18 $\rightarrow$ L17 \\
Qwen2.5-14B & 0.931 & 0.749 & -0.182 & L30 $\rightarrow$ L30 \\
Qwen2.5-14B-Instruct & 0.944 & 0.781 & -0.164 & L31 $\rightarrow$ L29 \\
Qwen2.5-3B & 0.882 & 0.688 & -0.193 & L27 $\rightarrow$ L23 \\
Qwen2.5-3B-Instruct & 0.896 & 0.695 & -0.200 & L30 $\rightarrow$ L26 \\
Qwen2.5-72B-Instruct & 0.955 & 0.812 & -0.144 & L58 $\rightarrow$ L59 \\
Qwen2.5-Math-1.5B & 0.905 & 0.703 & -0.202 & L20 $\rightarrow$ L16 \\
Qwen2.5-Math-1.5B-Instruct & 0.903 & 0.717 & -0.186 & L19 $\rightarrow$ L18 \\
Qwen2.5-Math-7B & 0.913 & 0.772 & -0.141 & L19 $\rightarrow$ L19 \\
Qwen2.5-Math-7B-Instruct & 0.921 & 0.785 & -0.136 & L19 $\rightarrow$ L21 \\
deepseek-math-7b-instruct & 0.867 & 0.696 & -0.171 & L18 $\rightarrow$ L15 \\
gemma-3-12b-it & 0.906 & 0.773 & -0.133 & L28 $\rightarrow$ L25 \\
gemma-3-27b-it & 0.928 & 0.840 & -0.088 & L36 $\rightarrow$ L33 \\
gemma-3-4b-it & 0.849 & 0.692 & -0.157 & L16 $\rightarrow$ L17 \\
gpt-oss-20b & 0.891 & 0.751 & -0.140 & L21 $\rightarrow$ L16 \\
\bottomrule
\end{tabular}
\caption{Generative Momentum (UMWP): Collapse of Latent F1 across sequence length.}
\label{tab:momentum_collapse}
\end{table}