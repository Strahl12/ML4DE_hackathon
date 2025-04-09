## Simple CNN trained on data only
import numpy as np
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pdb


# Load the data
data = np.load('/Users/johnparry/Projects/m4dl/hackathon-4-2025/ML4DE_hackathon/data/ks_training.npy')
#pdb.set_trace()
test_data = np.load('/Users/johnparry/Projects/m4dl/hackathon-4-2025/ML4DE_hackathon/data/ks_truth.npy')

# flip data
#data=np.transpose(data_train)
data_tensor= torch.from_numpy(data).float()
# reshape datatensor to include channels
data_tensor=torch.reshape(data_tensor,(data_tensor.shape[0],1,data_tensor.shape[1]))
# Create a TensorDataset
# Note that input_features and labels must match on the length of the first dimension.
dataset = TensorDataset(data_tensor[0:-1,:,:], data_tensor[1:,:,:]-data_tensor[0:-1,:,:])


def data_generator(path: str):
    data = np.load(path)
    pass


# Parameters


learning_rate=0.01
meshsize=2048
datasize=101
batch_size=20

# Initialise the dataloader

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
#pdb.set_trace()


# Set up CNN
conv1_size=6
conv2_size=12
kernel_size=10


hidden_size = [10,1] 

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # For 1D convolutions, the shape is [batch_size, num_channels, width]
        self.conv1 = nn.Conv1d(1, conv1_size, kernel_size)  # First Conv layer
        self.conv2 = nn.Conv1d(conv1_size, conv2_size, kernel_size) # Second Conv layer
        self.fc1 = nn.Linear((meshsize-2*(kernel_size-1))*conv2_size , 2048)                    # Fully connected layer 1
        self.fc2 = nn.Linear(2048, meshsize)                            # Fully connected layer 2 (output)
        self.linear1 = nn.Conv1d(1, hidden_size[0], 5, padding = 2, padding_mode = 'circular')
        self.linear2 = nn.Linear(hidden_size[0],hidden_size[1])

    def forward(self, x):
#        print(f" ##### {x.shape}")
        x = self.linear1(x)
        x = x.permute(0, 2, 1)
        x = F.tanh(x)
        x = self.linear2(x)
        x = x.permute(0, 2, 1)
        #x = tanh(par1*a+par2*b)
        
        #x = F.relu(self.conv1(x))  # Conv1 + ReLU
        #pdb.set_trace()
        #x = F.relu(self.conv2(x))  # Conv2 + ReLU
#        x = x.view(x.size(0), -1)
        #x = F.relu(self.fc1(x))               # FC1 + ReLU
        #x = self.fc2(x)                       # Output layer
        return x

# Initialize the network, optimizer, and loss function
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# Device configuration (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)



x=torch.rand(2,1,meshsize)
model(x)

losses = []
# Training the model


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Ensure target is float and matches the output shape
        target = target.float().unsqueeze(1) if target.dim() == 1 else target.float()

        # Compute the loss (MSE Loss)
        loss = criterion(output, target)

        # Backward pass and optimize weights
        loss.backward()
        optimizer.step()

        # Accumulate loss over the epoch (scaled by batch size)
        running_loss += loss.item() * data.size(0)

        # Optionally, print loss every 10 batches
        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}')
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')
    
    # Compute the average loss for the epoch
    avg_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch {epoch} Average Loss: {avg_loss:.6f}')
    return avg_loss

# Train the network and record the average loss for each epoch
num_epochs = 1000
epoch_loss_history = []

for epoch in range(1, num_epochs + 1):
    avg_loss = train(model, device, train_loader, optimizer, epoch)
    epoch_loss_history.append(avg_loss)

# Generate output data




