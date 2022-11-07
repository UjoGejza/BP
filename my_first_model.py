import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# custom data sets
class CustomDataset(Dataset): 
    def __init__(self, data, labels):

   
        self.data = torch.tensor(data)
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


train_data = [[0,0,0,0,0],
        [1,0,0,0,0],
        [1,1,0,0,0],
        [0,0,1,0,0],
        [1,0,0,0,1],
        [0,1,0,1,0],
        [1,1,0,1,1],
        [0,0,0,0,1],
        [0,1,0,0,1],
        [1,1,1,1,1]]
train_labels = [0,1,3,3,2,4,6,1,3,9] 

test_data = [[0,1,0,0,0],
        [0,0,0,1,1],
        [1,0,1,0,1],
        [1,0,0,1,1],
        [0,0,0,1,0],
        [1,1,1,0,0],
        [0,0,1,1,1],
        [1,0,0,1,0],
        [1,1,1,1,0]]
test_labels = [2,3,5,4,2,6,6,3,8] 

training_data = CustomDataset(train_data, train_labels)
test_data = CustomDataset(test_data, test_labels)

# Create data loaders.
train_dataloader = DataLoader(training_data)
test_dataloader = DataLoader(test_data)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            #nn.Linear(10, 10),
            #nn.ReLU(),
            #nn.Linear(10, 10)
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


model = NeuralNetwork().to(device)
print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for (X, y) in dataloader:
        print(X.size())
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

train(train_dataloader, model, loss_fn, optimizer)
