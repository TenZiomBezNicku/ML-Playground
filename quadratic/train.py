import torch
import torch.nn.functional as F
from torch import nn

model = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))

x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
y = torch.tensor([[4.0], [1.0], [0.0], [1.0], [4.0]])

print(model(x))

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for step in range(50000):
    optimizer.zero_grad()

    loss = F.mse_loss(model(x), y)

    loss.backward()

    optimizer.step()

    if step % 1000 == 0:
        print(f"step {step}")

print(model(x))