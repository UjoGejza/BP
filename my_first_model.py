import torch
from torch.utils.data import Dataset
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# custom data sets

# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class CustomDataset(Dataset): 
    def __init__(self, data, labels):

   
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)

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
        [1,1,1,1,0],
        [1,1,1,1,1],#from train
        [0,0,1,0,0],]#from train
test_labels = [2,3,5,4,2,6,6,3,8,9,3] 


training_data = CustomDataset(train_data, train_labels)
test_data = CustomDataset(test_data, test_labels)

# Create data loaders.
train_dataloader = DataLoader(dataset=training_data, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, shuffle=False)

examples = iter(train_dataloader)
samples, labels = examples.next()
print(samples.shape, labels.shape)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

#hyper parameters
input_size = 5
hidden_size = 10
num_classes = 10
num_epochs = 500
#batch_size = 1
learning_rate = 0.005

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)
        

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        #no softmax here, because we use CE loss later
        return out


model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)
#print(model)

loss_fn = nn.CrossEntropyLoss()
print(model.parameters())
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

def train():
    n_total_steps = len(train_dataloader)
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_dataloader):
            x = x.reshape(-1, 5).to(device)
            y = y.to(device)

            outputs = model(x)
            loss = loss_fn(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
                print(f'epoch {epoch+1} / {num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')
                test()

def test():
    n_correct = 0
    n_samples = 0
    for x, y in test_dataloader:
        x = x.reshape(-1, 5).to(device)
        y = y.to(device)
        outputs = model(x)

        #value, index
        _, predictions = torch.max(outputs, 1)
        n_samples += y.shape[0]
        n_correct += (predictions == y).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Test finished, accuracy = {acc:.2f}%')

test()
train()




"""def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for (X, y) in train_dataloader:
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
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")"""

