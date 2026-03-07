"""ResNet-specific wrapper around the generic HPO worker."""

from hpo_worker import main


if __name__ == "__main__":
    main(
        default_model="resnet",
        include_model=False,
        description="ResNet Fashion-MNIST HPO worker",
    )


