import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from lipschitz.data import datasets, data_loader, preprocessors
from lipschitz.io_functions import state_dict, save_results
from lipschitz.io_functions.parser import dictionary_str
from lipschitz.training.metrics import METRICS
from lipschitz.training.tracker import MetricTracker

# import datasets

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

eval_parser = argparse.ArgumentParser()
add_arg = eval_parser.add_argument
add_arg("sd_name", choices=state_dict.NAMES)
add_arg("-d", "--dataset", default="CIFAR10", choices=datasets.NAMES)
add_arg("-b", "--batch-size", type=int, default=256)
add_arg("-pkw", "--preprocessor-kwargs", type=dictionary_str, default="{}")
add_arg("-n", "--sample-size", type=int, default=100)
add_arg("-met", "--metrics-name", choices=METRICS, default="Lipschitz")
add_arg("--test", action="store_true")
add_arg("--save", action="store_true")
add_arg("-exp", "--experiment-name", default=None)


class TTAEvaluator:
    def __init__(self,
                 model=None,
                 metrics=None,
                 gpu_processing=lambda x: x,
                 sample_number: int = 100):
        self.model = model
        self.metrics = metrics

        self.gpu_processing = gpu_processing  # <= Test Time Augmentation here.
        self.sample_number = sample_number

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __call__(self, *args, **kwargs):
        return self.evaluate(*args, **kwargs)

    def evaluate(self, dl: DataLoader):
        tracker = MetricTracker(self.metrics)
        self.model.eval()
        with torch.no_grad():
            for x_batch, y_batch in dl:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                x_batch = x_batch[None]
                x_batch = x_batch.repeat((self.sample_number, 1, 1, 1, 1))
                x_batch = x_batch.view(-1, *x_batch.shape[2:])

                x_batch = self.gpu_processing(x_batch)
                predictions = self.model(x_batch)

                n = self.sample_number
                predictions = predictions.view(n, -1, predictions.shape[-1])
                predictions = predictions.mean(dim=0)  # Average over samples

                tracker.add_batch(predictions, y_batch)

        return tracker.results_dict


def evaluate_with_augmentation(*args):
    a = eval_parser.parse_args(args)
    print(f"Arguments: {a}")

    ds = datasets.get(a.dataset)
    # d = ds.get_data(use_test_data=a.test, training_size=a.training_size)
    d = ds.get_data(use_test_data=a.test)
    dl = data_loader.get_data_loader(d, a.batch_size)
    print(f"Loaded {len(dl.train)}/{len(dl.eval)} train/eval batches.")

    preprocess = preprocessors.Concatenate(
        preprocessors.Center(ds.channel_means),
        preprocessors.get(**a.preprocessor_kwargs),
    )

    model = state_dict.load_trained_model(a.sd_name)
    model.to(DEVICE)

    print("\nEvaluating...")
    e = TTAEvaluator(model, METRICS[a.metrics_name], gpu_processing=preprocess,
                     sample_number=a.sample_size)
    output = {
        "command": " ".join(sys.argv),
        "arguments": vars(a),
        "filename": Path(__file__).name,
        "results": {"train": e(dl.train), "eval": e(dl.eval)},
    }

    if a.save:
        save_results(output)
    else:
        print(f"Results:")
        print(yaml.safe_dump(output).replace("  ", "    "))

