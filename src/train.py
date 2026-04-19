import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from utils import load_dataset
from model import CurlClassifier
from export_mobile import export_to_torchscript

def train_model(epochs=30, batch_size=32, lr=0.001):
    print("Loading dataset...")
    X, y = load_dataset('data')
    
    if len(X) == 0:
        print("Error: No data found in 'data/' directory.")
        return

    # Transpose X to (N, C, L) for Conv1d: (batch, channels, seq_len)
    X = np.transpose(X, (0, 2, 1))
    
    # Normalization
    mean = X.mean(axis=(0, 2))
    std = X.std(axis=(0, 2))
    print(f"Normalization Stats:")
    print(f"Mean: {mean.tolist()}")
    print(f"Std:  {std.tolist()}")
    
    X = (X - mean[None, :, None]) / (std[None, :, None] + 1e-7)
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    
    dataset = TensorDataset(X_tensor, y_tensor)
    
    # Split into train and validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    model = CurlClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()
            
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('models_saved', exist_ok=True)
            export_to_torchscript(model, 'models_saved/curl_classifier.pt')
            print(f"** Saved best model to models_saved/curl_classifier.pt (Val Loss: {best_val_loss:.4f})")

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs}: "
                  f"Loss: {train_loss/len(train_loader):.4f}, Acc: {100.*correct/total:.2f}% | "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {100.*val_correct/val_total:.2f}%")

    print(f"Training finished. Best Val Loss: {best_val_loss:.4f}")

if __name__ == "__main__":
    train_model(epochs=30)
