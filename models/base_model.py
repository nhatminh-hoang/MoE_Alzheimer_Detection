import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # Define your model layers here
        self.fc1 = nn.Linear(10, 5)  # Example: Input size 10, output size 5

    def forward(self, x):
        # Define the forward pass
        x = F.relu(self.fc1(x))  # Example: ReLU activation
        return x
    
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop_out)
        self.norm = nn.BatchNorm1d(hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)        
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, stride=1, padding='same')
        self.norm1 = nn.BatchNorm1d(hidden_size)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size*2, kernel_size=3, stride=1, padding='same')
        self.norm2 = nn.BatchNorm1d(hidden_size*2)
        self.conv3 = nn.Conv1d(hidden_size*2, hidden_size*4, kernel_size=3, stride=1, padding='same')
        self.norm3 = nn.BatchNorm1d(hidden_size*4)
        self.pool = nn.AdaptiveMaxPool1d(output_size=8)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(hidden_size*4, hidden_size*4)
        self.fc2 = nn.Linear(hidden_size*4, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add channel dimension if necessary
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(0, 2, 1)
            
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.squeeze(-1)  # Remove the last dimension
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x