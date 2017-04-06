from IPython.core.magic import Magics, magics_class, cell_magic

import argparse
import shlex

from flowly.ipy import notify


@magics_class
class FlowlyMagics(Magics):
    @cell_magic
    def notify(self, line, cell):
        parser = argparse.ArgumentParser()
        parser.add_argument('--tag')
        parser.add_argument('message')
        args = parser.parse_args(shlex.split(line))

        self.shell.run_cell(cell)
        notify(args.message, tag=args.tag)


get_ipython().register_magics(FlowlyMagics)  # noqa
