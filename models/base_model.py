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
    def __init__(self, input_size, hidden_size, output_size, drop_out, train=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(drop_out)
        self.norm = nn.BatchNorm1d(hidden_size)

        self.training = train

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