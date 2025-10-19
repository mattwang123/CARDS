import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt

class UncertaintyExplorer:
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
    
    def get_representation(self, prompt: str):
        """Get the last token representation of input prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Last token of input prompt - move to CPU for numpy conversion
            last_token_repr = outputs.last_hidden_state[0, -1, :].cpu().numpy()
        
        return last_token_repr
    
    def compare_prompts(self, prompts: list, labels: list = None):
        """Compare representations across multiple prompts"""
        representations = []
        
        for i, prompt in enumerate(prompts):
            repr = self.get_representation(prompt)
            representations.append(repr)
            label = labels[i] if labels else "unknown"
            print(f"[{label.upper()}] Prompt: '{prompt[:50]}...'")
            print(f"Representation shape: {repr.shape}")
            print(f"Norm: {np.linalg.norm(repr):.3f}")
            
            # Compare with previous representations
            if i > 0:
                for j in range(i):
                    similarity = np.dot(repr, representations[j]) / (
                        np.linalg.norm(repr) * np.linalg.norm(representations[j])
                    )
                    prev_label = labels[j] if labels else "unknown"
                    print(f"  Similarity with prompt {j+1} ({prev_label}): {similarity:.3f}")
            print()
        
        return np.stack(representations)
    
    def visualize_clustering(self, representations, labels):
        """Simple 2D visualization of representation clustering"""
        from sklearn.decomposition import PCA
        
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(representations)
        
        plt.figure(figsize=(10, 8))
        
        # Create labels for matched pairs
        sufficient_count = 0
        insufficient_count = 0
        
        # Plot points with different colors and markers
        for i, (point, label) in enumerate(zip(reduced, labels)):
            color = 'blue' if label == 'sufficient' else 'red'
            marker = 'o' if label == 'sufficient' else '^'
            
            # Create paired labels
            if label == 'sufficient':
                sufficient_count += 1
                point_label = f'{sufficient_count}_t'
            else:
                insufficient_count += 1
                point_label = f'{insufficient_count}_f'
                
            plt.scatter(point[0], point[1], c=color, marker=marker, s=100, alpha=0.7)
            plt.annotate(point_label, (point[0], point[1]), xytext=(5, 5), 
                        textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Add legend
        plt.scatter([], [], c='blue', marker='o', s=100, label='Sufficient (_t)', alpha=0.7)
        plt.scatter([], [], c='red', marker='^', s=100, label='Insufficient (_f)', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Add grid and labels
        plt.grid(True, alpha=0.3)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
        plt.title('Representation Clustering: Sufficient vs Insufficient\n(1_t/1_f = polynomials, 2_t/2_f = numbers, 3_t/3_f = books)', fontsize=14)
        
        # Print additional info
        print(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, PC2={pca.explained_variance_ratio_[1]:.1%}")
        print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.1%}")
        
        plt.tight_layout()
        plt.savefig('uncertainty_clustering.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Plot saved as 'uncertainty_clustering.png'")

def main():
    # Model configuration
    model_name = "tbs17/MathBERT"
    
    # Initialize explorer
    explorer = UncertaintyExplorer(model_name)
    
    # Test prompts
    prompts = [
        # SUFFICIENT (complete mathematical statements)
        "Find all monic polynomials ( f ) with integer coefficients satisfying the following condition: There exists a positive integer ( N ) such that for every prime ( p > N ), ( p ) divides ( 2(f(p))! + 1 ).",
        
        "Find all natural numbers ( n ) for which ( 1^{\phi(n)} + 2^{\phi(n)} + \ldots + n^{\phi(n)} ) is coprime with ( n ).",
        
        "Determine the number of ways to arrange 5 distinct books on a shelf.",
        
        # INSUFFICIENT (missing key information)
        "Find all monic polynomials f satisfying the condition.",
        
        "Find all natural numbers n for which the sum is coprime with n.",
        
        "Determine the number of ways to arrange the books."
    ]
    
    labels = ['sufficient'] * 3 + ['insufficient'] * 3
    
    # Run experiment
    representations = explorer.compare_prompts(prompts, labels)
    explorer.visualize_clustering(representations, labels)

if __name__ == "__main__":
    main()