import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GNNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        # He init (default for linear layers in pytorch)
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv**2)
        if self.bias is not None:
            self.bias.data.normal_(0, stdv**2)

    def forward(self, x, adj):
        # TODO CODE HERE
        pass

class GNN(nn.Module):
    def __init__(self, in_features, hidden_size, out_features, adjacency_matrix):
        super(GNN, self).__init__()
        self.layer1 = GNNLayer(in_features, hidden_size)
        self.layer2 = GNNLayer(hidden_size, out_features)
        self.adjacency_matrix = adjacency_matrix

    def forward(self, x):
        x = F.relu(self.layer1(x, self.adjacency_matrix))
        x = self.layer2(x, self.adjacency_matrix)
        return x

def add_self_loops(adj):
    assert adj.size(0) == adj.size(1), "Adjacency matrix must be square"
    # TODO CODE HERE
    pass

def normalize_adj(adj):
    """ Symmetric normalization """
    # TODO CODE HERE (square root could be inf for 0 degree nodes, set those to 0)
    pass

### Mock data
x = torch.randn(5, 3)  # Node features (5 nodes, 3 features each)
adj = torch.tensor([[ 0, 1, 0, 1, 0],
                     [1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1],
                     [1, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0]], dtype=torch.float32)  # Adjacency matrix (5 nodes)

# Node 1 has the same class as nodes 2 and 4
y = torch.tensor([1, 1, 0, 1, 0], dtype=torch.long)  # Labels (5 nodes)
###

### Preprocess adjacency matrix
print("Original adjacency matrix\n", adj)
# Add self loops
# NORMalize adjacency matrix using symmetric normalization
adj = add_self_loops(adj)
print("...with self loops\n", adj)
adj = normalize_adj(adj)
print("Normalized adjacency matrix\n", adj)
input("Press any key to continue")
###

### Training
model = GNN(in_features=3, hidden_size=16, out_features=2, adjacency_matrix=adj)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(1000):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    optimizer.step()
###

# print accuracy
with torch.no_grad():
    output = model(x)
    predicted = torch.argmax(output, dim=1)
    accuracy = (predicted == y).float().mean()
    print(f'Accuracy: {accuracy.item() * 100:.2f}%')