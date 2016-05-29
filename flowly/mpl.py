from ._base import lit, side_effect
import matplotlib.pyplot as plt

# TODO: add proper __all__

gca = lit(plt.gca)

xlabel = side_effect(plt.xlabel)
ylabel = side_effect(plt.ylabel)

xlim = side_effect(plt.xlim)
ylim = side_effect(plt.ylim)

title = side_effect(plt.title)
suptitle = side_effect(plt.suptitle)

plot = side_effect(plt.plot)
legend = side_effect(plt.legend)

def _labels(x, y, title=None):
    plt.xlabel(x)
    plt.ylabel(y)

    if title is not None:
        plt.title(title)

labels = side_effect(_labels)
