import os
from torchvision import datasets, transforms


def get_datasets(data_dir: str):
    os.makedirs(data_dir, exist_ok=True)

    # Pure, standard MNIST 1-channel normalization
    mnist_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    print("[Dataset] Loading pure MNIST (1-channel, 28x28)...")
    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=mnist_transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=mnist_transform
    )

    metadata = {"num_classes": 10, "in_channels": 1, "input_size": 28}

    return train_dataset, test_dataset, metadata
