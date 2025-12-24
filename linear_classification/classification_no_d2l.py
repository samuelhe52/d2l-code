import torch
from torch import nn
from utils.classfication import train, get_dataloader

class MNISTSoftmax(nn.Module):
    """
    The Fashion-MNIST classifier using softmax regression.
    """
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(num_inputs, num_outputs)
        )
        
    def forward(self, X):
        return self.net(X)
        
if __name__ == "__main__":
    batch_size = 512
    num_epochs = 10
    lr = 0.02
    num_inputs = 28 * 28
    num_outputs = 10
    model = MNISTSoftmax(num_inputs, num_outputs)
    train_loader = get_dataloader(batch_size, data_root='../data')
    train(model, train_loader, num_epochs, lr)