import argparse
from itertools import product

from lipschitz.io_functions.parser import dictionary_str
from lipschitz.scripts.basic.time_batch import time_batch

DEFAULT_RANGE = "{'model.base_width': [16, 32, 64, 128]}"

parser = argparse.ArgumentParser()
add_arg = parser.add_argument
add_arg("-u", "--updates", type=dictionary_str, default="{}")
add_arg("-r", "--ranges", type=dictionary_str, default=DEFAULT_RANGE)
add_arg("-n", "--sample_number", type=int, default=100)


def main():
    a = parser.parse_args()
    print(f"Arguments: {a}")
    time_batch_all(a)


def time_batch_all(a):
    for chosen_values in product(*a.ranges.values()):
        print(f"Choose values {chosen_values} for {list(a.ranges.keys())}.")
        for key, value in zip(a.ranges.keys(), chosen_values):
            a.updates[key] = value

        time_batch(a.updates, a.sample_number)


if __name__ == "__main__":
    main()
