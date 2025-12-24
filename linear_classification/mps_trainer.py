from d2l import torch as d2l
import torch

class MPSTrainer(d2l.Trainer):
    """
    Trainer adapted for MPS support.
    """
    def __init__(self, max_epochs, num_devices=0, gradient_clip_val=0):
        # Don't call super().__init__; we reimplement init logic
        self.save_hyperparameters()
        devices = []

        # Prefer CUDA if available
        if torch.cuda.device_count() > 0:
            devices = [torch.device(f'cuda:{i}')
                       for i in range(torch.cuda.device_count())]
        # Otherwise, try MPS
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices = [torch.device("mps")]
        # Fallback to CPU
        else:
            devices = [torch.device("cpu")]

        self.gpus = devices[:num_devices] if num_devices > 0 else []
