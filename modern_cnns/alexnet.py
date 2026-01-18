import torch
from torch import nn
from utils.classfication import train, get_dataloader
from utils import TrainingLogger

class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 4096), nn.ReLU(),
            nn.Dropout(), # Default dropout rate of 0.5
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, X):
        return self.net(X)

if __name__ == "__main__":
    batch_size = 128
    num_epochs = 10
    lr = 0.01
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    model = AlexNet().to(device)
    dataloader = get_dataloader(batch_size, resize=224, data_root='data/')
    val_dataloader = get_dataloader(batch_size, train=False, resize=224, data_root='data/')
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    init_fn = torch.nn.init.kaiming_uniform_
    
    hparams = {
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'lr': lr,
        'init_fn': init_fn.__name__,
    }
    
    logger = TrainingLogger(
        log_path='logs/alexnet_experiment.json',
        hparams=hparams
    )
    
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init_fn(m.weight)
            
    model.apply(init_weights)
    
    train(model, dataloader=dataloader, num_epochs=num_epochs,
          lr=lr, loss_fn=nn.CrossEntropyLoss(),
          optimizer=optimizer, save_path='models/alexnet.pt',
          logger=logger, val_dataloader=val_dataloader, device=device)
    
    logger.summary()
    logger.save()
    