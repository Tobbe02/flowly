from IPython import get_ipython

from contextlib import contextmanager


try:
    from collections import ChainMap

except ImportError:
    from chainmap import ChainMap


def scope(args, source):
    """Create isolates scopes inside an ipython notebook.

    Usage::

        # In[1]
        all_scopes = 20

        # In[2]
        %%scope a
        a = 42  # can only be seen in a

        # In[3]
        %%scope b
        b = 21  # can only be seen in b

        # In[4]
        %%scope a
        print(all_scopes)  # prints 20
        print(a)  # prints 42
        print(b)  # fails with an exception

        # In[5]
        %%scope b
        print(all_scopes)  # prints 20
        print(b)  # prints 21
        print(a)  # fails with an exception

    """
    scope_name = args.strip()
    shell = get_ipython().kernel.shell

    if not scope_name:
        shell.run_cell(source)
        return None

    user_ns = shell.user_ns
    scopes = user_ns.setdefault('__scopes__', {})

    if scope_name not in scopes:
        scopes[scope_name] = ChainMap({}, user_ns)

    shell.user_ns = scopes[scope_name]

    try:
        shell.run_cell(source)

    finally:
        shell.user_ns = user_ns


get_ipython().register_magic_function(scope, 'cell')

