import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal
from collections import deque
from scipy.signal import hilbert
from scipy.linalg import expm
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import torch

# Import hippocampal classes directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "hippo_test", 
    "tests/test_hippocampal_formation.py"
)
hippo_module = importlib.util.module_from_spec(spec)

# Execute only the class definitions (before the unittest.TestCase)
with open('tests/test_hippocampal_formation.py', 'r') as f:
    content = f.read()
    # Extract only the class definitions, not the test class
    class_defs = content.split('class TestHippocampalFormation')[0]
    exec(class_defs)



class MNISTHippocampalMemory:
    def __init__(self, n_samples=100):
        self.hippo = HippocampalFormation(
            n_place_cells=50, 
            n_time_cells=20, 
            n_grid_cells=30
        )
        self.interpolator = TemporalMemoryInterpolator()
        
        # Load MNIST
        print("Loading MNIST dataset...")
        mnist_data = datasets.MNIST(
            root='./tests/data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        
        # Sample subset
        indices = np.random.choice(len(mnist_data), n_samples, replace=False)
        self.images = []
        self.labels = []
        self.features = []
        
        for idx in indices:
            img, label = mnist_data[idx]
            self.images.append(img.squeeze().numpy())
            self.labels.append(label)
            # Flatten to feature vector
            feat = img.view(-1).numpy()
            self.features.append(feat)
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.features = np.array(self.features)
        
        # Project to 2D for spatial mapping
        print("Computing 2D projection...")
        pca = PCA(n_components=2)
        self.spatial_coords = pca.fit_transform(self.features)
        
        # Normalize to reasonable range
        self.spatial_coords = (self.spatial_coords - self.spatial_coords.mean(axis=0))
        self.spatial_coords = self.spatial_coords / (self.spatial_coords.std(axis=0) + 1e-8) * 5
        
        print(f"Loaded {n_samples} MNIST samples")
        print(f"Spatial coords range: X[{self.spatial_coords[:, 0].min():.2f}, {self.spatial_coords[:, 0].max():.2f}], Y[{self.spatial_coords[:, 1].min():.2f}, {self.spatial_coords[:, 1].max():.2f}]")
        
        # Tracking
        self.stored_indices = []
        self.current_idx = 0
    
    def store_memory(self, idx):
        if idx >= len(self.images):
            return None
        
        # Update spatial state
        location = self.spatial_coords[idx]
        self.hippo.update_spatial_state(location, dt=0.1)
        
        # Create episodic memory
        features = self.features[idx]
        label = self.labels[idx]
        mem_id = f"digit_{label}_idx_{idx}"
        
        self.hippo.create_episodic_memory(
            memory_id=mem_id,
            event_id=f"event_{idx}",
            features=features,
            associated_experts=[f"digit_{label}"]
        )
        
        self.stored_indices.append(idx)
        return mem_id
    
    def query_similar(self, query_idx, k=5):
        if query_idx >= len(self.features):
            return []
        
        query_features = self.features[query_idx]
        query_location = self.spatial_coords[query_idx]
        
        similar = self.hippo.retrieve_similar_memories(
            query_features,
            location=query_location,
            k=k
        )
        
        return similar


def live_visualization_demo():
    print("\n" + "="*60)
    print("MNIST Hippocampal Memory - Live Visualization")
    print("="*60)
    
    # Create system
    memory_system = MNISTHippocampalMemory(n_samples=50)
    
    # Setup figure
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Axes
    ax_spatial = fig.add_subplot(gs[:2, :2])  # Spatial map
    ax_current = fig.add_subplot(gs[0, 2])    # Current image
    ax_place = fig.add_subplot(gs[0, 3])       # Place cells
    ax_grid = fig.add_subplot(gs[1, 2])        # Grid cells
    ax_time = fig.add_subplot(gs[1, 3])        # Time cells
    ax_similar = fig.add_subplot(gs[2, :2])    # Similar memories
    ax_stats = fig.add_subplot(gs[2, 2:])      # Statistics
    
    # Color map for digits
    digit_colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def update(frame):
        if frame >= len(memory_system.images):
            return
        
        # Store new memory
        mem_id = memory_system.store_memory(frame)
        
        # Clear axes
        ax_spatial.clear()
        ax_current.clear()
        ax_place.clear()
        ax_grid.clear()
        ax_time.clear()
        ax_similar.clear()
        ax_stats.clear()
        
        # 1. Spatial map
        ax_spatial.set_title('Spatial Memory Map (PCA Projection)', fontweight='bold')
        
        # Plot all stored memories
        for idx in memory_system.stored_indices:
            loc = memory_system.spatial_coords[idx]
            label = memory_system.labels[idx]
            color = digit_colors[label]
            
            if idx == frame:
                # Highlight current
                ax_spatial.scatter(loc[0], loc[1], c=[color], s=300, 
                                 edgecolors='red', linewidths=3, alpha=0.9, marker='*')
                ax_spatial.annotate(f'{label}', (loc[0], loc[1]), 
                                  fontsize=12, ha='center', va='center', 
                                  color='white', fontweight='bold')
            else:
                ax_spatial.scatter(loc[0], loc[1], c=[color], s=100, 
                                 alpha=0.7, edgecolors='black', linewidths=1)
                ax_spatial.annotate(f'{label}', (loc[0], loc[1]), 
                                  fontsize=8, ha='center', va='center', color='white')
        
        # Draw connections for recent memories
        if len(memory_system.stored_indices) > 1:
            recent = memory_system.stored_indices[-min(5, len(memory_system.stored_indices)):]
            for i in range(len(recent)-1):
                loc1 = memory_system.spatial_coords[recent[i]]
                loc2 = memory_system.spatial_coords[recent[i+1]]
                ax_spatial.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], 
                              'gray', alpha=0.3, linewidth=1, linestyle='--')
        
        ax_spatial.set_xlabel('PCA Component 1')
        ax_spatial.set_ylabel('PCA Component 2')
        ax_spatial.set_aspect('equal')
        ax_spatial.grid(True, alpha=0.3)
        
        # 2. Current image
        ax_current.imshow(memory_system.images[frame], cmap='gray')
        ax_current.set_title(f'Current: Digit {memory_system.labels[frame]}', fontweight='bold')
        ax_current.axis('off')
        
        # 3. Place cell activity
        spatial_ctx = memory_system.hippo.get_spatial_context()
        place_activity = spatial_ctx['place_cells']
        ax_place.bar(range(len(place_activity)), place_activity, color='blue', alpha=0.7)
        ax_place.set_title('Place Cells', fontsize=10)
        ax_place.set_ylabel('Rate (Hz)', fontsize=8)
        ax_place.tick_params(labelsize=7)
        ax_place.grid(True, alpha=0.3)
        
        # 4. Grid cell activity
        grid_activity = spatial_ctx['grid_cells']
        ax_grid.bar(range(len(grid_activity)), grid_activity, color='green', alpha=0.7)
        ax_grid.set_title('Grid Cells', fontsize=10)
        ax_grid.set_ylabel('Rate (Hz)', fontsize=8)
        ax_grid.tick_params(labelsize=7)
        ax_grid.grid(True, alpha=0.3)
        
        # 5. Time cell activity
        temporal_ctx = memory_system.hippo.get_temporal_context()
        time_activity = temporal_ctx['time_cells']
        ax_time.bar(range(len(time_activity)), time_activity, color='red', alpha=0.7)
        ax_time.set_title('Time Cells', fontsize=10)
        ax_time.set_ylabel('Rate (Hz)', fontsize=8)
        ax_time.tick_params(labelsize=7)
        ax_time.grid(True, alpha=0.3)
        
        # 6. Similar memories
        if len(memory_system.stored_indices) > 1:
            similar = memory_system.query_similar(frame, k=min(5, len(memory_system.stored_indices)))
            
            for i, (mem_id, score) in enumerate(similar[:5]):
                # Extract index from memory ID
                try:
                    idx_str = mem_id.split('_idx_')[1]
                    similar_idx = int(idx_str)
                    img = memory_system.images[similar_idx]
                    label = memory_system.labels[similar_idx]
                    
                    ax_pos = ax_similar.inset_axes([i*0.19 + 0.02, 0.1, 0.18, 0.8])
                    ax_pos.imshow(img, cmap='gray')
                    ax_pos.set_title(f'{label}\n{score:.3f}', fontsize=8)
                    ax_pos.axis('off')
                except:
                    pass
        
        ax_similar.set_title('Similar Memories (Retrieval)', fontweight='bold')
        ax_similar.axis('off')
        
        # 7. Statistics
        stats_text = f"""
        Hippocampal Memory Statistics
        
        Memories Stored: {len(memory_system.hippo.episodic_memories)}
        Current Memory: {frame + 1}/{len(memory_system.images)}
        
        Digit Distribution:
        {np.bincount(memory_system.labels[memory_system.stored_indices], minlength=10)}
        
        Spatial Locations: {len(memory_system.hippo.spatial_locations)}
        Cognitive Map Edges: {len(memory_system.hippo.cognitive_map)}
        
        Place Cells Active: {np.sum(place_activity > 0)}/{len(place_activity)}
        Grid Cells Active: {np.sum(grid_activity > 0)}/{len(grid_activity)}
        Time Cells Active: {np.sum(time_activity > 0)}/{len(time_activity)}
        
        Theta Phase: {spatial_ctx['theta_phase']:.2f} rad
        """
        
        ax_stats.text(0.05, 0.5, stats_text, fontsize=9, family='monospace',
                     verticalalignment='center')
        ax_stats.axis('off')
        
        plt.suptitle(f'Live MNIST Hippocampal Memory - Frame {frame+1}', 
                    fontsize=14, fontweight='bold')
    
    # Create animation
    print("\nStarting live visualization...")
    print("This will store MNIST digits in hippocampal memory in real-time")
    
    anim = FuncAnimation(fig, update, frames=len(memory_system.images), 
                        interval=500, repeat=False)
    
    # Save animation
    output_path = 'tests/artifacts/mnist_hippocampal_live.gif'
    print(f"\nSaving animation to {output_path}...")
    anim.save(output_path, writer='pillow', fps=2, dpi=100)
    print(f"Animation saved!")
    
    # Also show final state
    plt.figure(figsize=(12, 8))
    
    # Plot all memories colored by digit
    for label in range(10):
        mask = memory_system.labels[memory_system.stored_indices] == label
        coords = memory_system.spatial_coords[memory_system.stored_indices][mask]
        if len(coords) > 0:
            plt.scatter(coords[:, 0], coords[:, 1], 
                       c=[digit_colors[label]], s=150, 
                       label=f'Digit {label}', alpha=0.7,
                       edgecolors='black', linewidths=1)
    
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Final Memory Spatial Organization by Digit Class')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    
    final_path = 'tests/artifacts/mnist_hippocampal_final.png'
    plt.savefig(final_path, dpi=150, bbox_inches='tight')
    print(f"Final visualization saved to {final_path}")
    
    plt.show()


if __name__ == '__main__':
    live_visualization_demo()
