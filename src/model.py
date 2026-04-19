import torch
import torch.nn as nn
import torch.nn.functional as F

class CurlClassifier(nn.Module):
    def __init__(self, input_channels=6, num_classes=2):
        """
        CNN 1D for multi-channel time-series classification.
        Input shape: (batch, channels, seq_len)
        """
        super(CurlClassifier, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        
        # Max Pooling
        self.pool = nn.MaxPool1d(2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Fully connected layers
        # After two MaxPool1d(2), a seq_len of 100 becomes 100/2/2 = 25
        self.fc1 = nn.Linear(64 * 25, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected block
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == "__main__":
    # Test the model with a dummy input
    model = CurlClassifier()
    dummy_input = torch.randn(1, 6, 100)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("Model initialized and tested successfully.")
