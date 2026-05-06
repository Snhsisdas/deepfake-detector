import torch
import torch.nn as nn
import torchvision.models as models

class DeepFakeDetector(nn.Module):
    def __init__(self, hidden_dim=256, lstm_layers=1, num_classes=2, dropout=0.5):
        super(DeepFakeDetector, self).__init__()
        
        # Load a pre-trained lightweight CNN (MobileNetV2)
        # Suitable for a 4GB VRAM GPU
        mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Remove the final classification layer (classifier)
        # features will output (batch_size, 1280, 7, 7) for 224x224 input
        self.feature_extractor = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # Output: (batch_size, 1280, 1, 1)
        
        # LSTM for temporal modeling
        # Input to LSTM is the feature vector size from CNN (1280 for MobileNetV2)
        self.lstm = nn.LSTM(input_size=1280, 
                            hidden_size=hidden_dim, 
                            num_layers=lstm_layers, 
                            batch_first=True, 
                            dropout=dropout if lstm_layers > 1 else 0)
        
        # Final classification head
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, channels, height, width)
        batch_size, seq_length, c, h, w = x.size()
        
        # Reshape to (batch_size * seq_length, channels, height, width) 
        # to pass through the CNN in one go
        x = x.view(batch_size * seq_length, c, h, w)
        
        # Extract features
        features = self.feature_extractor(x)
        features = self.avgpool(features)
        # Flatten
        features = features.view(features.size(0), -1) # Shape: (batch_size * seq_length, 1280)
        
        # Reshape back to sequence for LSTM
        # Shape: (batch_size, seq_length, 1280)
        features = features.view(batch_size, seq_length, -1)
        
        # Pass through LSTM
        # lstm_out shape: (batch_size, seq_length, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(features)
        
        # Take the output of the last time step for classification
        last_step_out = lstm_out[:, -1, :] # Shape: (batch_size, hidden_dim)
        
        # Final classifier
        out = self.fc1(last_step_out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out) # Shape: (batch_size, num_classes)
        
        return out

if __name__ == "__main__":
    # Test the model
    model = DeepFakeDetector()
    # Dummy input: batch_size=2, seq_length=5, channels=3, 224x224
    dummy_input = torch.randn(2, 5, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Should be (2, 2)
