from model import QModel
import torch


nn = QModel(4, 2)

q = nn(torch.randn(1,4))
maxes = torch.argmax(q).item()


print(q)
print(maxes)