
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import numpy as np

class EmbeddingVisualizer:
    def __init__(self, embeddings: torch.Tensor, labels=None, types=None):
        """
        embeddings: [N, D] tensor
        labels: list of strings (names of rules)
        types: list of ints/strings (types of rules)
        """
        self.embeddings = embeddings.detach().cpu().numpy()
        self.labels = labels
        self.types = types
        
    def plot_3d(self, save_path=None, block=True):
        if self.embeddings.shape[1] < 3:
            print("Embedding dimension < 3, cannot perform 3D PCA.")
            return

        pca = PCA(n_components=3)
        components = pca.fit_transform(self.embeddings)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Color map based on types if available
        colors = 'b'
        legend_elements = []
        
        if self.types is not None:
            # Simple color mapping
            unique_types = sorted(list(set(self.types)))
            # mapped to ints if strings
            type_to_int = {t: i for i, t in enumerate(unique_types)}
            mapped_types = [type_to_int[t] for t in self.types]
            
            scatter = ax.scatter(components[:, 0], components[:, 1], components[:, 2], 
                               c=mapped_types, cmap='tab10', marker='o', s=50, alpha=0.6)
            
            # Create a legend
            from matplotlib.lines import Line2D
            cmap = plt.get_cmap('tab10')
            for t in unique_types:
                idx = type_to_int[t]
                color = cmap(idx/10) if len(unique_types) <= 10 else cmap(idx/len(unique_types))
                # Note: scatter cmap normalization is complex, for simple legend:
                legend_elements.append(Line2D([0], [0], marker='o', color='w', label=str(t),
                                            markerfacecolor=scatter.to_rgba(idx), markersize=10))
            
            ax.legend(handles=legend_elements, loc='best')

        else:
            ax.scatter(components[:, 0], components[:, 1], components[:, 2], c='b', marker='o')
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('GILP Embedding Space (3D PCA)')
        
        if self.labels:
             # Annotate points. 
             # To avoid clutter, we only annotate Conjectures or if total points are small
             for i, txt in enumerate(self.labels):
                 is_target = False
                 if self.types:
                     t = self.types[i]
                     if isinstance(t, str) and 'conjecture' in t.lower(): is_target = True
                     if isinstance(t, int) and t == 1: is_target = True # content convention
                 
                 if is_target or len(self.labels) < 50:
                    ax.text(components[i, 0], components[i, 1], components[i, 2], txt, size=9)

        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        print("Displaying plot... (Close window to continue if blocking)")
        try:
            plt.show(block=block)
        except Exception as e:
            print(f"Could not display plot: {e}")

