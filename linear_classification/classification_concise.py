import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
from mps_trainer import MPSTrainer

class FashionMNISTSoftmax(d2l.Classifier):
    """
    The Fashion-MNIST classifier using softmax regression.
	"""
    def __init__(self, num_outputs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
			nn.Flatten(), nn.LazyLinear(num_outputs)
		)

    def forward(self, X):
        return self.net(X)


if __name__ == "__main__":
    data = d2l.FashionMNIST(batch_size=1024)
    model = FashionMNISTSoftmax(num_outputs=10, lr=0.1)
    trainer = d2l.Trainer(max_epochs=10)
    trainer.fit(model, data)
    d2l.plt.show()
