import sys

from lipschitz import data
from lipschitz.io_functions.parser import dictionary_str


CIFAR10_kwargs = {"name": "CIFAR-10"}


def download_data(dataset_kwargs: str = "{'name': 'CIFAR-10'}"):
    data_kwargs = dictionary_str(dataset_kwargs)
    print(f"Loading dataset with kwargs: {data_kwargs}")

    ds = data.get_dataset(**data_kwargs)
    ds.download()


if __name__ == "__main__":
    dataset_kwargs = sys.argv[1] if len(sys.argv) > 1 else str(CIFAR10_kwargs)
    print(f"Using dataset kwargs: {dataset_kwargs}")
    download_data(dataset_kwargs)
