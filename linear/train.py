import torch
import torch.nn.functional as F
from torch import nn

model = nn.Linear(1, 1)

x = torch.tensor([[-2.0], [-1.0], [0.0], [1.0], [2.0]])
y = torch.tensor([[5.0], [3.0], [1.0], [-1.0], [-3.0]])

print(model(x))

optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

for step in range(10000):
    optimizer.zero_grad()

    loss = F.mse_loss(model(x), y)

    loss.backward()

    optimizer.step()

    if step % 100 == 0:
        print(f"step {step}")

print(model(x))