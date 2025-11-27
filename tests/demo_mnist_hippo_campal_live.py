import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import sys
import os
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.hippocampal import HippocampalFormation

class MNISTHippocampalMemory:
    def __init__(self, n_samples=100):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # FIXED: Set feature_dim=784 to match MNIST flattened images (28x28)
        self.hippo = HippocampalFormation(
            spatial_dimensions=2,
            n_place_cells=100, 
            n_time_cells=20, 
            n_grid_cells=50,
            feature_dim=784,  # <--- CRITICAL FIX
            device=self.device
        )
        
        # Load MNIST
        print("Loading MNIST...")
        try:
            # Check if data exists, download if not
            mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        except Exception as e:
            print(f"Download failed: {e}. Creating fake data.")
            # Fake data for environments without internet
            self.images = np.random.rand(n_samples, 28, 28)
            self.labels = np.random.randint(0, 10, n_samples)
            self.features = np.random.randn(n_samples, 784)
            self.spatial_coords = np.random.randn(n_samples, 2) * 5
            self.stored_indices = []
            return

        # Sample subset
        indices = np.random.choice(len(mnist_data), n_samples, replace=False)
        self.images = []
        self.labels = []
        self.features = []
        
        for idx in indices:
            img, label = mnist_data[idx]
            self.images.append(img.squeeze().numpy())
            self.labels.append(label)
            # Flatten 28x28 -> 784
            self.features.append(img.view(-1).numpy())
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.features = np.array(self.features)
        
        # PCA Projection for 2D spatial map
        print("Computing 2D spatial projection...")
        pca = PCA(n_components=2)
        self.spatial_coords = pca.fit_transform(self.features)
        # Normalize to reasonable range [-5, 5]
        self.spatial_coords = (self.spatial_coords - self.spatial_coords.mean(0)) / (self.spatial_coords.std(0) + 1e-6) * 5
        
        self.stored_indices = []

    def store_memory(self, idx):
        if idx >= len(self.images): return
        
        # 1. Update Spatial State
        loc = torch.tensor(self.spatial_coords[idx], dtype=torch.float32, device=self.device)
        self.hippo.update_spatial_state(loc)
        
        # 2. Store Memory
        # Features must match feature_dim (784)
        feat = torch.tensor(self.features[idx], dtype=torch.float32, device=self.device)
        self.hippo.create_episodic_memory(f"mem_{idx}", f"evt_{idx}", feat)
        
        self.stored_indices.append(idx)

def live_visualization_demo():
    print("Initializing Demo...")
    # Use fewer samples for speed in demo
    memory_system = MNISTHippocampalMemory(n_samples=20)
    
    if len(memory_system.images) == 0:
        print("No data loaded. Exiting.")
        return

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig)
    ax_map = fig.add_subplot(gs[0, 0])
    ax_act = fig.add_subplot(gs[0, 1])
    ax_img = fig.add_subplot(gs[1, 0])
    ax_stat = fig.add_subplot(gs[1, 1])
    
    def update(frame):
        # Stop if we run out of data
        if frame >= len(memory_system.images):
            return
            
        memory_system.store_memory(frame)
        
        # 1. Map
        ax_map.clear()
        ax_map.set_title("Spatial Map (PCA)")
        # Plot place centers (Background)
        centers = memory_system.hippo.place_centers.cpu().numpy()
        ax_map.scatter(centers[:,0], centers[:,1], c='gray', alpha=0.3, s=10, label="Place Cells")
        
        # Plot stored memories
        if memory_system.stored_indices:
            locs = memory_system.spatial_coords[memory_system.stored_indices]
            ax_map.scatter(locs[:,0], locs[:,1], c='blue', s=30, label="Memories")
            # Highlight current
            curr = locs[-1]
            ax_map.scatter(curr[0], curr[1], c='red', s=100, marker='*', label="Current")
        ax_map.legend(loc='lower right', fontsize='small')
            
        # 2. Activity (Place Cells)
        ax_act.clear()
        ctx = memory_system.hippo.get_spatial_context()
        # Convert tensor to numpy
        activity = ctx['place_cells'].cpu().numpy()
        ax_act.bar(range(len(activity)), activity, color='orange')
        ax_act.set_title("Place Cell Activity")
        ax_act.set_ylim(0, 25)
        
        # 3. Image
        ax_img.clear()
        ax_img.imshow(memory_system.images[frame], cmap='gray')
        ax_img.set_title(f"Input: Digit {memory_system.labels[frame]}")
        ax_img.axis('off')
        
        # 4. Stats
        ax_stat.clear()
        stats_text = (
            f"Step: {frame+1}/{len(memory_system.images)}\n"
            f"Memories: {memory_system.hippo.memory_count}\n"
            f"Active Place Cells: {(activity > 0.1).sum()}\n"
            f"Device: {memory_system.device}"
        )
        ax_stat.text(0.1, 0.5, stats_text, fontsize=12, family='monospace')
        ax_stat.axis('off')

    print("Generating animation...")
    # Frames must match n_samples or be fewer
    frames = len(memory_system.images)
    anim = FuncAnimation(fig, update, frames=frames, interval=200, repeat=False)
    
    output_dir = 'tests/artifacts'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'mnist_live_gpu.gif')
    
    try:
        anim.save(output_path, writer='pillow', fps=5)
        print(f"✅ Animation saved to {output_path}")
    except Exception as e:
        print(f"❌ Could not save animation: {e}")

if __name__ == '__main__':
    live_visualization_demo()