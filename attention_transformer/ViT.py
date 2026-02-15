import torch
from torch import nn, Tensor
from typing import Optional

from utils.attn import MultiheadAttentionWithValidLens
from utils.data import fashion_mnist
from utils.training import (
    TrainingConfig, TrainingLogger,
    ClassificationTrainer
)
from utils.io import load_model
from utils.enc_dec import Encoder


class PatchEmbedding(nn.Module):
    """
    Split the image into patches and then embed them.
    
    Args:
        img_size: The size of the input image (assumed to be square).
        patch_size: The size of each patch (assumed to be square).
        embed_dim: The dimension of the embedding for each patch.
    """
    def __init__(self, img_size: int,
                 patch_size: int, embed_dim: int = 32):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.conv = nn.LazyConv2d(
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, X: Tensor) -> Tensor:
        # X shape: (batch_size, channels, height, width)
        # Output shape: (batch_size, num_patches, embed_dim)
        return self.conv(X).flatten(2).transpose(1, 2)

        
class ViTMLP(nn.Module):
    """
    A simple MLP for the ViT model.
    
    Args:
        embed_dim: The dimension of the input embedding.
        mlp_hidden_dim: The dimension of the hidden layer in the MLP.
        dropout: The dropout rate.
    """
    def __init__(self, embed_dim: int, mlp_hidden_dim: int, 
                 dropout: float = 0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, X: Tensor) -> Tensor:
        return self.mlp(X)


class ViTBlock(nn.Module):
    """
    A single block of the ViT model, consisting of multi-head attention and an MLP.
    
    Args:
        embed_dim: The dimension of the input embedding.
        num_heads: The number of attention heads.
        mlp_hidden_dim: The dimension of the hidden layer in the MLP.
        dropout: The dropout rate.
    """
    def __init__(self, embed_dim: int, num_heads: int, 
                 mlp_hidden_dim: int, dropout: float = 0.5):
        super().__init__()
        self.attn = MultiheadAttentionWithValidLens(embed_dim, num_heads, dropout)
        self.mlp = ViTMLP(embed_dim, mlp_hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(
        self, X: Tensor,
        valid_lens: Optional[Tensor] = None) -> Tensor:
        # Multi-head attention with residual connection
        X = self.norm1(X)
        Y = X + self.attn(X, X, X, valid_lens)[0]

        # MLP with residual connection
        return Y + self.mlp(self.norm2(Y))

        
class ViT(Encoder):
    """
    The Vision Transformer (ViT) model for image classification.

    Args:
        img_size: The size of the input image (assumed to be square).
        patch_size: The size of each patch (assumed to be square).
        embed_dim: The dimension of the embedding for each patch.
        num_heads: The number of attention heads.
        mlp_hidden_dim: The dimension of the hidden layer in the MLP.
        num_blocks: The number of ViT blocks to stack.
        num_classes: The number of output classes for classification.
        embed_dropout: The dropout rate for the patch embedding.
        blk_dropout: The dropout rate for the ViT blocks.
    """
    def __init__(self, img_size: int, patch_size: int, embed_dim: int,
                 num_heads: int, mlp_hidden_dim: int, num_blocks: int,
                 num_classes: int, embed_dropout: float, blk_dropout: float):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, embed_dim)
        self.dropout = nn.Dropout(embed_dropout)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        seq_len = self.patch_embedding.num_patches + 1  # +1 for the class token
        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, embed_dim))
        self.blocks = nn.Sequential(
            *[ViTBlock(embed_dim, num_heads, mlp_hidden_dim, blk_dropout) 
              for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, X: Tensor) -> Tensor:
        # X shape: (batch_size, channels, height, width)
        batch_size = X.shape[0]
        X = self.patch_embedding(X)  # (batch_size, num_patches, embed_dim)
        class_token = self.class_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        X = torch.cat((class_token, X), dim=1)  # (batch_size, num_patches + 1, embed_dim)
        X = X + self.pos_embedding  # Add positional embedding
        X = self.dropout(X)
        X = self.blocks(X)  # (batch_size, num_patches + 1, embed_dim)
        return self.head(X[:, 0, :]) # The first token is the class token


if __name__ == "__main__":
    hparams = {
        'batch_size': 128,
        'num_epochs': 20,
        'lr': 1e-3,
        'img_size': 96,
        'patch_size': 16,
        'embed_dim': 256,
        'num_heads': 8,
        'mlp_hidden_dim': 512,
        'num_blocks': 6,
        'embed_dropout': 0.1,
        'blk_dropout': 0.1,
    }

    train_loader = fashion_mnist(
        hparams['batch_size'],
        resize=hparams['img_size'],
        data_root='data/',
    )
    val_loader = fashion_mnist(
        hparams['batch_size'],
        train=False,
        resize=hparams['img_size'],
        data_root='data/',
    )

    model = ViT(
        img_size=hparams['img_size'],
        patch_size=hparams['patch_size'],
        embed_dim=hparams['embed_dim'],
        num_heads=hparams['num_heads'],
        mlp_hidden_dim=hparams['mlp_hidden_dim'],
        num_blocks=hparams['num_blocks'],
        num_classes=10,
        embed_dropout=hparams['embed_dropout'],
        blk_dropout=hparams['blk_dropout'],
    )

    logger = TrainingLogger(
        log_path='logs/vit_experiment.json',
        hparams=hparams,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=hparams['lr'])

    config = TrainingConfig(
        num_epochs=hparams['num_epochs'],
        lr=hparams['lr'],
        optimizer=optimizer,
        save_path='models/vit',
        logger=logger,
    )

    def init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    model(torch.randn(1, 1, hparams['img_size'], hparams['img_size']))
    model.apply(init_weights)

    trainer = ClassificationTrainer(model, train_loader, val_loader, config)
    trainer.train()
    logger.summary()
    