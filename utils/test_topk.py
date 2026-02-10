import torch

a = torch.tensor([[0.1, 0.3, 0.2, 0.4],
                  [0.5, 0.2, 0.2, 0.1],
                  [0.25, 0.25, 0.25, 0.25]])

print(a.shape)
top2_values, top2_indices = torch.topk(a, k=2, dim=1)
print("Top-2 values:\n", top2_values)
print("Top-2 indices:\n", top2_indices)