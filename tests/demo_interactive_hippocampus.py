import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Literal
from collections import deque
            n_place_cells=50, 
            n_time_cells=20, 
            n_grid_cells=30
        )
        
        print("Loading MNIST dataset...")
        mnist_data = datasets.MNIST(
            root='./tests/data',
            train=True,
            download=True,
            transform=transforms.ToTensor()
        )
        
        indices = np.random.choice(len(mnist_data), n_samples, replace=False)
        self.images = []
        self.labels = []
        self.features = []
        
        for idx in indices:
            img, label = mnist_data[idx]
            self.images.append(img.squeeze().numpy())
            self.labels.append(label)
            feat = img.view(-1).numpy()
            self.features.append(feat)
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.features = np.array(self.features)
        
        print("Computing spatial projection...")
        pca = PCA(n_components=2)
        self.spatial_coords = pca.fit_transform(self.features)
        self.spatial_coords = (self.spatial_coords - self.spatial_coords.mean(axis=0))
        self.spatial_coords = self.spatial_coords / (self.spatial_coords.std(axis=0) + 1e-8) * 5
        
        self.stored_indices = []
        self.current_idx = 0
        
        print(f"\nLoaded {n_samples} MNIST samples")
        print(f"Press 'n' for next memory, 's' to skip 5, 'a' for auto-play, 'q' to quit\n")
        
        self.digit_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        self.auto_play = False
    
    def store_next_memory(self):
        if self.current_idx >= len(self.images):
            print("\nAll memories stored!")
            return False
        
        location = self.spatial_coords[self.current_idx]
        self.hippo.update_spatial_state(location, dt=0.1)
        
        features = self.features[self.current_idx]
        label = self.labels[self.current_idx]
        mem_id = f"digit_{label}_idx_{self.current_idx}"
        
        self.hippo.create_episodic_memory(
            memory_id=mem_id,
            event_id=f"event_{self.current_idx}",
            features=features,
            associated_experts=[f"digit_{label}"]
        )
        
        self.stored_indices.append(self.current_idx)
        
        print(f"Stored memory {self.current_idx + 1}/{len(self.images)}: Digit {label} at location ({location[0]:.2f}, {location[1]:.2f})")
        
        self.current_idx += 1
        return True
    
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
    
    def visualize_current_state(self):
        plt.clf()
        
        fig = plt.gcf()
        gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)
        
        # 1. Spatial map
        ax_spatial = fig.add_subplot(gs[:2, :2])
        ax_spatial.set_title('Spatial Memory Map (Live)', fontweight='bold', fontsize=12)
        
        for idx in self.stored_indices:
            loc = self.spatial_coords[idx]
            label = self.labels[idx]
            color = self.digit_colors[label]
            
            if idx == self.current_idx - 1:
                ax_spatial.scatter(loc[0], loc[1], c=[color], s=350, 
                                 edgecolors='red', linewidths=4, alpha=1.0, marker='*', zorder=10)
                ax_spatial.annotate(f'{label}', (loc[0], loc[1]), 
                                  fontsize=14, ha='center', va='center', 
                                  color='white', fontweight='bold', zorder=11)
            else:
                alpha = 0.9 if idx in self.stored_indices[-10:] else 0.5
                ax_spatial.scatter(loc[0], loc[1], c=[color], s=120, 
                                 alpha=alpha, edgecolors='black', linewidths=1.5, zorder=5)
                ax_spatial.annotate(f'{label}', (loc[0], loc[1]), 
                                  fontsize=9, ha='center', va='center', 
                                  color='white', zorder=6)
        
        if len(self.stored_indices) > 1:
            recent = self.stored_indices[-min(8, len(self.stored_indices)):]
            for i in range(len(recent)-1):
                loc1 = self.spatial_coords[recent[i]]
                loc2 = self.spatial_coords[recent[i+1]]
                ax_spatial.plot([loc1[0], loc2[0]], [loc1[1], loc2[1]], 
                              'blue', alpha=0.4, linewidth=2, linestyle='--', zorder=1)
        
        ax_spatial.set_xlabel('PCA Component 1', fontsize=10)
        ax_spatial.set_ylabel('PCA Component 2', fontsize=10)
        ax_spatial.set_aspect('equal')
        ax_spatial.grid(True, alpha=0.3)
        
        # 2. Current image
        if self.current_idx > 0:
            ax_current = fig.add_subplot(gs[0, 2])
            current_img_idx = self.current_idx - 1
            ax_current.imshow(self.images[current_img_idx], cmap='gray')
            ax_current.set_title(f'Current: Digit {self.labels[current_img_idx]}', 
                               fontweight='bold', fontsize=11)
            ax_current.axis('off')
        
        # 3. Place cells
        ax_place = fig.add_subplot(gs[0, 3])
        spatial_ctx = self.hippo.get_spatial_context()
        place_activity = spatial_ctx['place_cells']
        ax_place.bar(range(len(place_activity)), place_activity, color='blue', alpha=0.7, width=1.0)
        ax_place.set_title('Place Cells', fontsize=10, fontweight='bold')
        ax_place.set_ylabel('Rate (Hz)', fontsize=9)
        ax_place.tick_params(labelsize=7)
        ax_place.grid(True, alpha=0.3, axis='y')
        ax_place.set_ylim(0, 25)
        
        # 4. Grid cells
        ax_grid = fig.add_subplot(gs[1, 2])
        grid_activity = spatial_ctx['grid_cells']
        ax_grid.bar(range(len(grid_activity)), grid_activity, color='green', alpha=0.7, width=1.0)
        ax_grid.set_title('Grid Cells', fontsize=10, fontweight='bold')
        ax_grid.set_ylabel('Rate (Hz)', fontsize=9)
        ax_grid.tick_params(labelsize=7)
        ax_grid.grid(True, alpha=0.3, axis='y')
        ax_grid.set_ylim(0, 30)
        
        # 5. Time cells
        ax_time = fig.add_subplot(gs[1, 3])
        temporal_ctx = self.hippo.get_temporal_context()
        time_activity = temporal_ctx['time_cells']
        ax_time.bar(range(len(time_activity)), time_activity, color='red', alpha=0.7, width=1.0)
        ax_time.set_title('Time Cells', fontsize=10, fontweight='bold')
        ax_time.set_ylabel('Rate (Hz)', fontsize=9)
        ax_time.tick_params(labelsize=7)
        ax_time.grid(True, alpha=0.3, axis='y')
        ax_time.set_ylim(0, 20)
        
        # 6. Similar memories
        ax_similar = fig.add_subplot(gs[2, :3])
        ax_similar.set_title('Similar Memories (Retrieved)', fontweight='bold', fontsize=11)
        
        if self.current_idx > 0 and len(self.stored_indices) > 1:
            current_img_idx = self.current_idx - 1
            similar = self.query_similar(current_img_idx, k=min(6, len(self.stored_indices)))
            
            for i, (mem_id, score) in enumerate(similar[:6]):
                try:
                    idx_str = mem_id.split('_idx_')[1]
                    similar_idx = int(idx_str)
                    img = self.images[similar_idx]
                    label = self.labels[similar_idx]
                    
                    ax_pos = ax_similar.inset_axes([i*0.155 + 0.025, 0.15, 0.14, 0.7])
                    ax_pos.imshow(img, cmap='gray')
                    ax_pos.set_title(f'{label} ({score:.3f})', fontsize=8)
                    ax_pos.axis('off')
                    
                    if similar_idx == current_img_idx:
                        for spine in ax_pos.spines.values():
                            spine.set_edgecolor('red')
                            spine.set_linewidth(3)
                except:
                    pass
        
        ax_similar.axis('off')
        
        # 7. Statistics
        ax_stats = fig.add_subplot(gs[2, 3])
        
        digit_dist = np.bincount(self.labels[self.stored_indices], minlength=10) if len(self.stored_indices) > 0 else np.zeros(10)
        
        stats_text = f"""LIVE STATS
        
Stored: {len(self.stored_indices)}/{len(self.images)}
Active: {self.current_idx}/{len(self.images)}

Digits: {digit_dist.astype(int).tolist()}

Places: {np.sum(place_activity > 0)}/{len(place_activity)}
Grids: {np.sum(grid_activity > 0)}/{len(grid_activity)}
Times: {np.sum(time_activity > 0)}/{len(time_activity)}

Cog Map: {len(self.hippo.cognitive_map)} edges

θ Phase: {spatial_ctx['theta_phase']:.2f} rad
        """
        
        ax_stats.text(0.05, 0.5, stats_text, fontsize=8.5, family='monospace',
                     verticalalignment='center')
        ax_stats.axis('off')
        
        status = "AUTO" if self.auto_play else "MANUAL"
        plt.suptitle(f'LIVE Hippocampal Memory Formation - {status} MODE', 
                    fontsize=13, fontweight='bold')
        
        plt.draw()
        plt.pause(0.001)
    
    def run_interactive(self):
        plt.ion()
        fig = plt.figure(figsize=(18, 10))
        
        def on_key(event):
            if event.key == 'n':
                self.store_next_memory()
                self.visualize_current_state()
            elif event.key == 's':
                for _ in range(5):
                    if not self.store_next_memory():
                        break
                self.visualize_current_state()
            elif event.key == 'a':
                self.auto_play = not self.auto_play
                print(f"\nAuto-play: {'ON' if self.auto_play else 'OFF'}")
            elif event.key == 'q':
                print("\nQuitting...")
                plt.close()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        self.visualize_current_state()
        
        print("\n" + "="*60)
        print("INTERACTIVE MODE - Controls:")
        print("  n = Store next memory")
        print("  s = Skip 5 memories")
        print("  a = Toggle auto-play")
        print("  q = Quit")
        print("="*60 + "\n")
        
        while plt.fignum_exists(fig.number):
            if self.auto_play and self.current_idx < len(self.images):
                if self.store_next_memory():
                    self.visualize_current_state()
                    plt.pause(0.3)
                else:
                    self.auto_play = False
            else:
                plt.pause(0.1)
        
        print("\nSession ended. Final statistics:")
        print(f"Total memories stored: {len(self.stored_indices)}")
        print(f"Cognitive map edges: {len(self.hippo.cognitive_map)}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("MNIST Hippocampal Memory - INTERACTIVE LIVE DEMO")
    print("="*60 + "\n")
    
    demo = InteractiveMNISTHippocampus(n_samples=50)
    demo.run_interactive()
