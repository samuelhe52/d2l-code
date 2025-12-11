from d2l import torch as d2l
import torch
from torch import nn
from matplotlib import pyplot as plt

class LinearRegressionD2L(d2l.Module):
    """
    A concise linear regression model
    """
    def __init__(self, lr):
        super().__init__()
        self.lr = lr
        self.net = nn.LazyLinear(1)
        self.net.weight.data.normal_(0, 0.01)
        self.net.bias.data.fill_(0)
        
    def forward(self, X):
        return self.net(X)

    def loss(self, y_hat, y):
        fn = nn.MSELoss()
        return fn(y_hat, y)
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
    
def get_w_and_b(model):
    w = model.net.weight.data
    b = model.net.bias.data
    return w, b

if __name__ == "__main__":
    train_nums = [5, 10, 100, 1000, 10000]
    true_w = torch.tensor([2.0, -3.4])
    true_b = 4.2
    
    w_errors = []
    b_errors = []
    for n in train_nums:
        print(f"Training with {n} examples")
        # Create a fresh model for each training run
        model = LinearRegressionD2L(lr=0.03)
        data = d2l.SyntheticRegressionData(
            true_w,
            true_b,
            noise=0.01,
            num_train=n,
            batch_size=min(32, n)
            )
        trainer = d2l.Trainer(max_epochs=5)
        trainer.fit(model, data)
        
        w, b = get_w_and_b(model)
        
        error_w = torch.norm(w - true_w)
        error_b = torch.abs(b - true_b)
        w_errors.append(error_w.item())
        b_errors.append(error_b.item())

    print("Weight errors:", w_errors)
    print("Bias errors:", b_errors)

    plt.figure()
    plt.plot(train_nums, w_errors, marker='o', label='Weight Error')
    plt.plot(train_nums, b_errors, marker='s', label='Bias Error')
    plt.xlabel('Number of training examples')
    plt.ylabel('Error')
    plt.legend()
    plt.xscale('log')
    plt.title('Parameter Error vs Training Set Size')
    plt.show()