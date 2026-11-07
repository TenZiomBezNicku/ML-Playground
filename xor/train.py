import torch
import torch.nn.functional as F
from torch import nn

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.Tanh(),
    nn.Linear(4, 1)
)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

print(torch.sigmoid(model(x)))

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for step in range(10000):
    optimizer.zero_grad()

    loss = F.binary_cross_entropy_with_logits(model(x), y)

    loss.backward()

    optimizer.step()

    if step % 1000 == 0:
        print(f"step {step}; loss: {loss.item()}")

print(torch.sigmoid(model(x)))