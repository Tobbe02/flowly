from ._base import lit, side_effect, wrapped
import matplotlib.pyplot as plt
import six

# TODO: add proper __all__

gca = lit(plt.gca)

xlabel = side_effect(plt.xlabel)
ylabel = side_effect(plt.ylabel)

xlim = side_effect(plt.xlim)
ylim = side_effect(plt.ylim)

title = side_effect(plt.title)
suptitle = side_effect(plt.suptitle)

legend = side_effect(plt.legend)

yscale = side_effect(plt.yscale)
xscale = side_effect(plt.xscale)


@wrapped
def plot(obj, *args, **kwargs):
    if len(args) == 0:
        args = (obj,)

    # TODO: handle strings for x/y

    plt.plot(*args, **kwargs)
    return obj


def _labels(x, y, title=None):
    plt.xlabel(x)
    plt.ylabel(y)

    if title is not None:
        plt.title(title)


labels = side_effect(_labels)


def _plot_params(**kwargs):
    if "figsize" in kwargs:
        plt.gcf().set_size_inches(*kwargs.pop("figsize"))

    for func, arg in kwargs.items():
        getattr(plt, func)(arg)


plot_params = side_effect(_plot_params)
