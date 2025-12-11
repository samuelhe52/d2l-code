from model_and_optim import LinearRegressionScratch
from d2l import torch as d2l
import torch
import matplotlib.pyplot as plt

model = LinearRegressionScratch(num_inputs=2, lr=0.03)
data = d2l.SyntheticRegressionData(w=torch.tensor([2, -3.4]), b=4.2, noise=0.01)
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)

# plt.show()

with torch.no_grad():
    print('Error of w:', model.w.reshape(1, -1) - data.w)
    print('Error of b:', model.b - data.b)