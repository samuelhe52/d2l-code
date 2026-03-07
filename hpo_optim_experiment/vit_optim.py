"""ViT-specific wrapper around the generic HPO worker."""

from hpo_worker import main


if __name__ == "__main__":
    main(
        default_model="vit",
        include_model=False,
        description="ViT Fashion-MNIST HPO worker",
    )