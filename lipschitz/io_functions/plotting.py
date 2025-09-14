from matplotlib import pyplot as plt


DEFAULT_COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
DEFAULT_MARKERS = "oDhsp*8PHX"

plt.rcParams.update({'font.size': 14})


def plot_line(x_values, y_values, alpha=0.5, **kwargs):
    sorted_value = zip(*sorted(zip(x_values, y_values)))
    return plt.plot(*sorted_value, alpha=alpha, **kwargs)


def outside_legend(**kwargs):
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', **kwargs)
