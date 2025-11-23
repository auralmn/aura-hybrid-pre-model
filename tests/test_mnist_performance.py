import unittest
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from training.hebbian_layer import OjaLayer
from training.optimized_whitener import OptimizedWhitener

class BioClassifier(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Hebbian Layer (Manually updated, not via autograd)
        self.whitener = OptimizedWhitener(dim=input_dim)
        self.oja = OjaLayer(input_dim=input_dim, n_components=hidden_dim, max_components=hidden_dim)
        
        # Readout Layer (Trained via Backprop)
        self.readout = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
        
    def forward(self, x_numpy):
        # 1. Unsupervised / Preprocessing (Numpy)
        # We assume x_numpy is a batch of numpy arrays
        
        features_list = []
        for i in range(x_numpy.shape[0]):
            x_w = self.whitener.transform(x_numpy[i])
            oja_out = self.oja.step(x_w)
            features_list.append(oja_out.y)
            
        features = np.stack(features_list)
        features_tensor = torch.FloatTensor(features)
        
        # 2. Supervised (PyTorch)
        x = self.dropout(features_tensor)
        x = self.relu(x) # Oja output is linear, add non-linearity
        logits = self.readout(x)
        return logits

class TestMNISTPerformance(unittest.TestCase):
    def load_mnist(self):
        try:
            from torchvision import datasets, transforms
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            os.makedirs(data_dir, exist_ok=True)
            
            train_dataset = datasets.MNIST(data_dir, train=True, download=True, 
                                         transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ]))
            test_dataset = datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.1307,), (0.3081,))
                                         ]))
            return train_dataset, test_dataset
        except ImportError:
            print("Torchvision not found. Skipping full MNIST test.")
            return None, None
        except Exception as e:
            print(f"Failed to load MNIST: {e}")
            return None, None

    def test_mnist_accuracy(self):
        print("\n--- Starting MNIST Performance Test ---")
        train_dataset, test_dataset = self.load_mnist()
        if train_dataset is None:
            self.skipTest("MNIST dataset could not be loaded")
            
        # Hyperparameters
        BATCH_SIZE = 64
        EPOCHS = 5
        LR = 0.0005  # Lowered from 0.001
        HIDDEN_DIM = 1024 # Increased for better capacity
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        
        # Initialize with lower Oja learning rate
        model = BioClassifier(hidden_dim=HIDDEN_DIM)
        model.oja.eta = 0.001 # Lowered from default 0.01
        
        optimizer = optim.Adam(model.readout.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Training BioClassifier (Oja + Linear) for {EPOCHS} epochs...")
        
        for epoch in range(EPOCHS):
            model.train()
            correct = 0
            total = 0
            running_loss = 0.0
            
            start_time = time.time()
            
            for batch_idx, (data, target) in enumerate(train_loader):
                # Flatten image
                x_numpy = data.view(-1, 784).numpy()
                
                # Debug: Check input
                if np.isnan(x_numpy).any():
                    print(f"NaN in input at batch {batch_idx}")
                    break
                
                # Forward pass with checks
                features_list = []
                for i in range(x_numpy.shape[0]):
                    x_w = model.whitener.transform(x_numpy[i])
                    if np.isnan(x_w).any():
                        print(f"NaN in whitener output at batch {batch_idx}, sample {i}")
                        break
                        
                    oja_out = model.oja.step(x_w)
                    if np.isnan(oja_out.y).any():
                        print(f"NaN in Oja output at batch {batch_idx}, sample {i}")
                        break
                    features_list.append(oja_out.y)
                
                if len(features_list) != x_numpy.shape[0]:
                    print("Aborting batch due to NaNs")
                    break
                    
                features = np.stack(features_list)
                features_tensor = torch.FloatTensor(features)
                
                optimizer.zero_grad()
                logits = model.readout(model.relu(model.dropout(features_tensor)))
                loss = criterion(logits, target)
                
                if torch.isnan(loss):
                    print(f"NaN loss at batch {batch_idx}")
                    break
                    
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch+1} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.4f}")
            
            epoch_acc = 100. * correct / total
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1} Finished. Avg Loss: {running_loss/len(train_loader):.4f}, Acc: {epoch_acc:.2f}%, Time: {elapsed:.1f}s")
            
            # Test Evaluation
            test_acc = self.evaluate(model, test_loader)
            print(f"Test Accuracy: {test_acc:.2f}%")
            
            if test_acc >= 95.0:
                print("Target accuracy reached!")
                break
                
        self.assertGreaterEqual(test_acc, 95.0, "Did not reach 95% accuracy")

    def evaluate(self, model, test_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in test_loader:
                x_numpy = data.view(-1, 784).numpy()
                logits = model(x_numpy)
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        return 100. * correct / total

if __name__ == '__main__':
    unittest.main()
