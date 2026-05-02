import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2
})


def _get_output_path(output_filename, output_dir="results"):
    if not output_filename:
        return None
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output_filename)

def plot_keyframes(video_path, key_frames_indices, output_filename=None, output_dir="results"):
    cap = cv2.VideoCapture(video_path)
    frames_to_show = []

    for idx in sorted(key_frames_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames_to_show.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames_to_show:
        print("No frames extracted.")
        return

    fig, axes = plt.subplots(1, len(frames_to_show), figsize=(15, 5))
    if len(frames_to_show) == 1:
        axes = [axes]
    for ax, frame, idx in zip(axes, frames_to_show, key_frames_indices):
        ax.imshow(frame)
        ax.axis('off')
        ax.set_title(f"{idx}", fontweight='bold')
    plt.tight_layout()
    
    out_path = _get_output_path(output_filename, output_dir)
    if out_path:
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved keyframes plot to {out_path}")
    plt.show()

def plot_feature_projection(features, labels=None, method='pca', title="Feature Projection", output_filename=None, output_dir="results"):
    if len(features) < 2:
        print("Not enough features to project.")
        return

    if method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
    elif method.lower() == 'tsne':
        perplexity = min(30, max(5, len(features) // 3))
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, max_iter=1000)
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")

    features_2d = reducer.fit_transform(features)

    fig, ax = plt.subplots(figsize=(10, 8))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        print(f"Plotting {len(unique_labels)} clusters...")
        
        import matplotlib.colors as mcolors
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                continue
                
            cluster_points = features_2d[labels == label]
            
            # Generate a unique color dynamically based on cluster index using HSV space
            hue = i / len(unique_labels)
            color = mcolors.hsv_to_rgb([hue, 0.85, 0.9])
            
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, 
                       label=f'Cluster {label}', alpha=0.6, s=40, edgecolors='black', linewidth=0.1)
            
            median = np.median(cluster_points, axis=0)
            
            ax.scatter(median[0], median[1], marker='X', s=250, c='red', edgecolors='black', linewidth=1.5, zorder=5)
            ax.annotate(f'C{label}', (median[0], median[1]), textcoords="offset points", xytext=(0,10), 
                        ha='center', fontsize=11, fontweight='bold', 
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=1, alpha=0.8))
    else:
        ax.scatter(features_2d[:, 0], features_2d[:, 1], alpha=0.7, s=40, c='#1f77b4', edgecolors='w', linewidth=0.5)

    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel(f"{method.upper()} Dimension 1", fontweight='bold')
    ax.set_ylabel(f"{method.upper()} Dimension 2", fontweight='bold')
    
    out_path = _get_output_path(output_filename, output_dir)
    if out_path:
        plt.savefig(out_path, bbox_inches='tight', dpi=300)
        print(f"Saved {method.upper()} plot to {out_path}")
    plt.show()

