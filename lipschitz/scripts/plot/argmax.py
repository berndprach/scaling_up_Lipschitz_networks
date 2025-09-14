import argparse
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt

from lipschitz.io_functions import plotting
from lipschitz.io_functions.parser import dictionary_str
from lipschitz.io_functions.plotting import plot_line
from lipschitz.io_functions.result_parsing import (
    get_results, similar, filter_results
)
from lipschitz.scripts.plot.results import add_legend, make_title

plt.rcParams.update({'font.size': 14})


RESULTS_FOLDER = Path("outputs", "results")
PLOT_FOLDER = Path("outputs", "result_plots")
PLOT_FOLDER.mkdir(exist_ok=True)

parser = argparse.ArgumentParser()
add_arg = parser.add_argument
add_arg("-c", "--constraints", type=dictionary_str, default="{'epochs': 100}")
add_arg("-s", "--split-by", type=str, default="arguments.model_name")
add_arg("-x", "--x-key", type=str, default="arguments.lr")
add_arg("-xs", "--x-scale", choices=["linear", "log"], default="linear")
add_arg("-y", "--y-key", type=str, default="results.eval.CRA(0.14)")
add_arg("-ys", "--y-scale", choices=["linear", "log"], default="linear")
add_arg("-g", "--goal", type=str, default=None)
add_arg("--min", action="store_true", default=False)


def main():
    a = parser.parse_args()
    print(f"Arguments: {a}")
    plot_arg_max(a)


def plot_arg_max(args):
    if args.goal is None:
        args.goal = args.y_key

    _best = min if args.min else max
    best = partial(_best, key=lambda r: r[args.goal])

    results = get_results()
    results = filter_results(results, args.constraints)
    results = filter_results(results, {}, [args.x_key, args.y_key, args.goal])

    split_values = set(r.get(args.split_by) for r in results)
    print(f"Splitting by {args.split_by} with values {split_values}.")

    # Get data:
    arg_max_run = {}
    all_xys = {}
    for sv in sorted(split_values, key=str):
        arg_max_run[sv] = {}
        split_results = [r for r in results if r.get(args.split_by) == sv]
        xs = set(r[args.x_key] for r in split_results)
        for x in xs:
            x_results = [r for r in split_results if similar(r[args.x_key], x)]
            print(f"s={sv}, {args.x_key}={x}: {len(x_results)} results.")
            arg_max_run[sv][x] = best(x_results)

        all_xs = [r[args.x_key] for r in split_results]
        all_ys = [r[args.y_key] for r in split_results]
        all_xys[sv] = (all_xs, all_ys)

    # Plot data:
    colors = iter(plotting.DEFAULT_COLORS)
    for split_value, arg_max_runs in arg_max_run.items():
        xs = list(arg_max_runs.keys())
        ys = [r[args.y_key] for r in arg_max_runs.values()]

        c = next(colors)
        plt.scatter(xs, ys, label=split_value, color=c)
        plot_line(xs, ys, color=c)
        d = {x: round(y, 3) for x, y in zip(xs, ys)}
        print(f"\"{split_value}-{args.y_key}\": {d}")

        plt.scatter(*all_xys[split_value], alpha=0.1, color=c, marker='x')

    plt.title(f"{args.y_key}")
    plt.xlabel(f"{args.x_key} ({args.x_scale})")
    plt.ylabel(args.y_key)

    plt.xscale(args.x_scale)
    plt.yscale(args.y_scale)

    add_legend(make_title(args.split_by), len(split_values))
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
