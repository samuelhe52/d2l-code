"""DenseNet-specific wrapper around the generic HPO worker."""

from hpo_worker import main


if __name__ == "__main__":
    main(
        default_model="densenet",
        include_model=False,
        description="DenseNet Fashion-MNIST HPO worker",
    )