import torch
from d2l import torch as d2l

class MLP(d2l.Classifier):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 num_hiddens,
                 lr=0.01):
        super().__init__()
        self.save_hyperparameters()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(num_inputs, num_hiddens),
            torch.nn.ReLU(),
            torch.nn.Linear(num_hiddens, num_outputs)
        )
        
model = MLP(num_inputs=784, num_outputs=10, num_hiddens=384, lr=0.01)
data = d2l.FashionMNIST(batch_size=256)
trainer = d2l.Trainer(max_epochs=15) 

if __name__ == "__main__":
    trainer.fit(model, data)
    d2l.plt.show()
