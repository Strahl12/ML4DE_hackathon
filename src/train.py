## Simple CNN trained on data only
import numpy as np
from torch import nn
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# Load the data
data_train = np.load('data/TrainingData/Train_KS/X1train.npy')
# flip data
data=np.transpose(data_train)
data_tensor= torch.from_numpy(data).float()
# reshape datatensor to include channels
data_tensor=torch.reshape(data_tensor,(data_tensor.shape[0],1,data_tensor.shape[1]))
# Create a TensorDataset
# Note that input_features and labels must match on the length of the first dimension.
dataset = TensorDataset(data_tensor[0:-1,:,:], data_tensor[1:,:,:]-data_tensor[0:-1,:,:])

# Parameters


learning_rate=0.01
meshsize=1024
datasize=1000-1
batch_size=100

# Initialise the dataloader

train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


# Set up CNN
conv1_size=6
conv2_size=12
kernel_size=20

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # For 1D convolutions, the shape is [batch_size, num_channels, width]
        self.conv1 = nn.Conv1d(1, conv1_size, kernel_size)  # First Conv layer
        self.conv2 = nn.Conv1d(conv1_size, conv2_size, kernel_size) # Second Conv layer
        self.fc1 = nn.Linear((meshsize-2*(kernel_size-1))*conv2_size , 2048)                    # Fully connected layer 1
        self.fc2 = nn.Linear(2048, meshsize)                            # Fully connected layer 2 (output)

    def forward(self, x):
        x = F.tanh(self.conv1(x))  # Conv1 + ReLU
        x = F.tanh(self.conv2(x))  # Conv2 + ReLU
        x = x.view(-1, (meshsize-2*(kernel_size-1))*conv2_size)            # Flatten the output from the conv layers
        x = F.tanh(self.fc1(x))               # FC1 + ReLU
        x = self.fc2(x)                       # Output layer
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


# Training the model
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Zero the gradient
        optimizer.zero_grad()

        # Forward pass
        output = model(data)

        # Ensure target has the same shape as the output (e.g., one-hot encoding for classification)
        target = target.float().unsqueeze(1) if target.dim() == 1 else target.float()

        # Compute the L2 loss (MSE Loss)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()

        # Optimize weights
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)}')
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}')

# Train the network
num_epochs=80
for epoch in range(1, num_epochs + 1):
    train(model, device, train_loader, optimizer, epoch)

# Simple check

print('Finished Training')
print(f'\nError in one datapoint {torch.norm(data_tensor[9,:,:]+model(data_tensor[9,:,:])-data_tensor[10,:,:])/torch.norm(data_tensor[10,:,:])}')
print(f'\nError in one datapoint {torch.norm(data_tensor[9,:,:]+model(data_tensor[9,:,:])-data_tensor[9,:,:])/torch.norm(data_tensor[10,:,:])}')
print(f'\nError in one datapoint {torch.norm(data_tensor[9,:,:]+model(data_tensor[9,:,:])-data_tensor[11,:,:])/torch.norm(data_tensor[10,:,:])}')

# Generate output data
output=torch.zeros(data_tensor.shape)
x=data_tensor[999,:,:]
for j in range(1000):
    output[j,:,:]=model(x)
    x=model(x.detach())

output=torch.reshape(output.detach(),(output.shape[0],data_tensor.shape[2]))

torch.transpose(output,0,1)
output_vector=(output.detach().numpy()).astype(np.float64)


# Save the NumPy array to an .npy file
np.save('data/Task1/KS_X1prediction.npy', output_vector)

# Plotting

plt.imshow(data_train, cmap='viridis')  # 'viridis' is a common color map for visualizations
plt.title('Training data')
plt.colorbar()  # Show a color bar to indicate value mapping to color
plt.show()


plt.imshow(output_vector, cmap='viridis')  # 'viridis' is a common color map for visualizations
plt.title('Predicted data')
plt.colorbar()  # Show a color bar to indicate value mapping to color
plt.show()