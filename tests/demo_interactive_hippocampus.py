import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import sys
import os
import torch
from torchvision import datasets, transforms
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.hippocampal import HippocampalFormation

class InteractiveMNISTHippocampus:
    def __init__(self, n_samples=50):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # FIXED: feature_dim=784
        self.hippo = HippocampalFormation(
            spatial_dimensions=2,
            n_place_cells=100, 
            n_time_cells=20, 
            n_grid_cells=50,
            feature_dim=784,
            device=self.device
        )
        
        print("Loading MNIST...")
        try:
            mnist_data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        except:
            print("Using fake data.")
            self.images = np.random.rand(n_samples, 28, 28)
            self.labels = np.random.randint(0, 10, n_samples)
            self.features = np.random.randn(n_samples, 784)
            self.spatial_coords = np.random.randn(n_samples, 2)
            self.stored_indices = []
            self.current_idx = 0
            self.digit_colors = plt.cm.tab10(np.linspace(0,1,10))
            self.auto_play = False
            return

        indices = np.random.choice(len(mnist_data), n_samples, replace=False)
        self.images = []
        self.labels = []
        self.features = []
        
        for idx in indices:
            img, label = mnist_data[idx]
            self.images.append(img.squeeze().numpy())
            self.labels.append(label)
            self.features.append(img.view(-1).numpy())
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.features = np.array(self.features)
        
        pca = PCA(n_components=2)
        self.spatial_coords = pca.fit_transform(self.features)
        self.spatial_coords = (self.spatial_coords - self.spatial_coords.mean(0)) / self.spatial_coords.std(0) * 5
        
        self.stored_indices = []
        self.current_idx = 0
        self.digit_colors = plt.cm.tab10(np.linspace(0, 1, 10))
        self.auto_play = False

    def store_next_memory(self):
        if self.current_idx >= len(self.images): return False
        
        loc = torch.tensor(self.spatial_coords[self.current_idx], dtype=torch.float32, device=self.device)
        feat = torch.tensor(self.features[self.current_idx], dtype=torch.float32, device=self.device)
        
        self.hippo.update_spatial_state(loc)
        self.hippo.create_episodic_memory(f"mem_{self.current_idx}", "evt", feat)
        
        self.stored_indices.append(self.current_idx)
        self.current_idx += 1
        return True

    def visualize_current_state(self):
        plt.clf()
        fig = plt.gcf()
        gs = GridSpec(2, 2, figure=fig)
        
        ax_map = fig.add_subplot(gs[0, 0])
        centers = self.hippo.place_centers.cpu().numpy()
        ax_map.scatter(centers[:,0], centers[:,1], c='gray', alpha=0.2, s=10)
        
        for idx in self.stored_indices:
            loc = self.spatial_coords[idx]
            color = self.digit_colors[self.labels[idx]]
            s = 100 if idx == self.current_idx - 1 else 30
            ax_map.scatter(loc[0], loc[1], color=color, s=s)
            
        ax_img = fig.add_subplot(gs[0, 1])
        if self.current_idx > 0:
            ax_img.imshow(self.images[self.current_idx-1], cmap='gray')
            
        plt.draw()
        plt.pause(0.001)

    def run_interactive(self):
        plt.ion()
        fig = plt.figure(figsize=(12, 8))
        
        def on_key(event):
            if event.key == 'n': self.store_next_memory(); self.visualize_current_state()
            elif event.key == 'q': plt.close()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        self.visualize_current_state()
        
        print("Press 'n' for next, 'q' to quit")
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

if __name__ == '__main__':
    demo = InteractiveMNISTHippocampus(n_samples=50)
    demo.run_interactive()