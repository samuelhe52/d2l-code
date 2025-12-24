import torch
from torch import nn
from utils.classfication import train, get_dataloader
from utils import TrainingLogger

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
    num_epochs = 20
    lr = 0.05
    num_inputs = 28 * 28
    num_outputs = 10
    model = MNISTSoftmax(num_inputs, num_outputs)
    train_loader = get_dataloader(batch_size, data_root='data')
    test_loader = get_dataloader(batch_size, train=False, data_root='data')

    logger = TrainingLogger(
        log_path='logs/mnist_softmax_experiment.json',
        hparams={
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'lr': lr
        }
    )
    
    train(model, train_loader, num_epochs, lr,
          test_dataloader=test_loader, logger=logger)
    
    logger.summary()
    # Persist log to disk
    logger.save()