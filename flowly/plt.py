from __future__ import print_function, division, absolute_import

import numpy as np
import matplotlib.pyplot as plt

from .tz import get_all_optional_items


class _Plotter(object):
    @staticmethod
    def bar(
        label=0, height=1, bottom=None, logy=False, rot=90,
        ylabel=None, xlabel=None, title=None
    ):
        set_labels = labels

        def impl(data):
            dl, dh, db = get_all_optional_items(label, height, bottom)(data)
            indices = np.arange(len(dl))

            plt.bar(indices, dh, bottom=db)
            plt.xticks(indices, dl, rotation=rot)

            if logy:
                plt.yscale('log')

            set_labels(x=xlabel, y=ylabel, title=title)

        return impl

    @staticmethod
    def scatter(x, y, s=None, c=None, **kwargs):
        def impl(data):
            dx, dy, ds, dc = get_all_optional_items(x, y, s, c)(data)

            plt.scatter(dx, dy, s=ds, c=dc, **kwargs)

        return impl

    @staticmethod
    def text(x, y, t, **kwargs):
        def impl(data):
            for item in data:
                plt.text(item[x], item[y], item[t], **kwargs)

        return impl


plot = _Plotter()


def labels(x=None, y=None, title=None):
    if x is not None:
        plt.xlabel(x)

    if y is not None:
        plt.ylabel(y)

    if title is not None:
        plt.title(title)
