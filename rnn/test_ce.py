# Testing nn.CrossEntropyLoss with different input shapes and target formats
import torch
import torch.nn as nn

if __name__ == "__main__":
    loss = nn.CrossEntropyLoss()
    
    # Case 1: Input shape (N, C) and target shape (N,)
    input1 = torch.randn(3, 5, requires_grad=True)  # 3 samples, 5 classes
    target1 = torch.tensor([1, 0, 4])  # Class indices
    output1 = loss(input1, target1)
    
    # Case 2: Input shape (N, C, d1) and target shape (N, d1)
    input2 = torch.randn(2, 4, 3, requires_grad=True)  # 2 samples, 4 classes, d1=3
    target2 = torch.tensor([[0, 1, 3], [2, 0, 1]])  # Class indices
    output2 = loss(input2, target2)