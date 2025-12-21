
"""
Epsilon Brain Visualizer

Visualizes the Parabolic Thinking Trajectory.
Projects 64-D vectors to 2-D using PCA to show the "Descent" and "Ascent".
"""

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
import numpy as np

def visualize_trajectory(trace, known_concepts, output_file="thought_trajectory.png"):
    """
    trace: List of (Label, Vector)
    known_concepts: Dict of {Name: Vector}
    """
    # 1. Gather all vectors for PCA fit
    all_vecs = []
    labels = []
    colors = []
    
    # Trace points
    for lbl, vec in trace:
        all_vecs.append(vec.detach().numpy())
        labels.append(lbl)
        colors.append('red') # Thought Path
        
    # Known concepts (Background stars)
    for name, vec in known_concepts.items():
        all_vecs.append(vec.detach().numpy())
        labels.append(name)
        colors.append('blue') # Truths
        
    X = np.array(all_vecs)
    
    # 2. PCA
    if len(X) < 2:
        print("Not enough points to plot.")
        return
        
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)
    
    # 3. Plot
    plt.figure(figsize=(10, 8))
    plt.style.use('dark_background')
    
    # Plot Trajectory Line
    trace_len = len(trace)
    plt.plot(X_r[:trace_len, 0], X_r[:trace_len, 1], 'r--', alpha=0.5, linewidth=2, label='Thinking Path')
    
    # Plot Points
    for i in range(len(X_r)):
        x, y = X_r[i]
        label = labels[i]
        color = colors[i]
        
        plt.scatter(x, y, c=color, s=100 if color=='red' else 50, alpha=0.8)
        
        # Annotate
        plt.text(x+0.02, y+0.02, label, color='white', fontsize=9)
        
    plt.title("Epsilon Parabolic Thought Trace")
    plt.xlabel("Principal Component 1 (Logic Axis 1)")
    plt.ylabel("Principal Component 2 (Logic Axis 2)")
    plt.legend()
    plt.grid(True, alpha=0.2)
    
    # Parabola Indicator (Simulated visual guide)
    # Ideally logic is center (0,0), atmosphere is outer rim.
    # We draw a circle to represent the 'Atmosphere' boundary if possible, 
    # but PCA distorts raw distance. We assume rough clustering.
    
    plt.savefig(output_file)
    print(f"✓ Visualization saved to {output_file}")
