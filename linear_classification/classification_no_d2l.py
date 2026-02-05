from torch import nn, Tensor
from utils.training import ClassificationTrainer
from utils.data import fashion_mnist
from utils import TrainingLogger, TrainingConfig

class MNISTSoftmax(nn.Module):
    """
    The Fashion-MNIST classifier using softmax regression.
    """
    def __init__(self, num_inputs: int, num_outputs: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(num_inputs, num_outputs)
        )
        
    def forward(self, X: Tensor) -> Tensor:
        return self.net(X)
        
if __name__ == "__main__":
    batch_size = 512
    num_epochs = 20
    lr = 0.05
    num_inputs = 28 * 28
    num_outputs = 10
    model = MNISTSoftmax(num_inputs, num_outputs)
    train_loader = fashion_mnist(batch_size, data_root='data')
    val_loader = fashion_mnist(batch_size, train=False, data_root='data')

    logger = TrainingLogger(
        log_path='logs/mnist_softmax_experiment.json',
        hparams={
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'lr': lr
        }
    )
    
    config = TrainingConfig(
        num_epochs=num_epochs,
        lr=lr,
        logger=logger,
    )
    
    trainer = ClassificationTrainer(model, train_loader, val_loader, config)
    trainer.train()
    
    logger.summary()
    # Persist log to disk
    logger.save()