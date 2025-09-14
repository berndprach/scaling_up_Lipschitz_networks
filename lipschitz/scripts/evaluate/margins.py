import argparse
from pathlib import Path

import yaml

from lipschitz import data
# import datasets

from lipschitz.io_functions import state_dict
from lipschitz.training import metrics


MARGIN_DIR = Path("outputs", "margins")
MARGIN_DIR.mkdir(parents=True, exist_ok=True)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-d", "--dataset", default="CIFAR10")
arg_parser.add_argument("-s", "--state-dict-name", choices=state_dict.NAMES)
arg_parser.add_argument("--test", action="store_true")


def evaluate_margins(*args):
    arg = arg_parser.parse_args(args)
    print(f"Arguments: {arg}")

    model = state_dict.load_trained_model(arg.state_dict_name)
    model.eval()

    results = state_dict.load_results(arg.state_dict_name)
    print(yaml.safe_dump(results))

    dl = data.get_loader(c["dataset"], c["preprocessing"])

    dataset = data.datasets.get(results["arguments"]["dataset"])
    dl = dataset.get_data_loader(use_test_data=arg.test)
    print(f"Loaded {len(dl.train)}/{len(dl.eval)} train/eval batches.")

    ms = [lipschitz.training.metrics.basic.Margin()]
    margins = {
        "train": evaluate(model, dl.train, ms).get_values()[0],
        "eval": evaluate(model, dl.eval, ms).get_values()[0],
    }
    print(
        f"Evaluated {len(margins['train'])} + {len(margins['eval'])} margins."
    )

    margin_fn = arg.state_dict_name.replace(".pt", ".yaml")
    if arg.test:
        margin_fn = margin_fn.replace(".yaml", "_test.yaml")
    with open(MARGIN_DIR / margin_fn, "w") as f:
        yaml.safe_dump(margins, f)
    print(f"Margins saved to {MARGIN_DIR / margin_fn}")
