from IPython import get_ipython


def scope(args, source):
    """Create isolates scopes inside an ipython notebook.

    Note, that any variable defined in the default scope, can be accessed from
    the child scopes.
    However, variables defined inside the child scopes cannot be seen from the
    default scope.

    Usage::

        %%scope a
        a = 42  # can only be seen in a

        %%scope b
        b = 21  # can only be seen in b

        %%scope a
        print(a)  # prints 42
        print(b)  # fails with an exception

        %%scope b
        print(b)  # prints 21
        print(a)  # fails with an exception

    """
    scope_name = args.strip()
    shell = get_ipython().kernel.shell

    if not scope_name:
        shell.run_cell(source)
        return None

    user_ns = shell.user_ns
    shell.user_ns = user_ns.setdefault('__scopes__', {}).setdefault(scope_name, {})

    try:
        shell.run_cell(source)

    finally:
        shell.user_ns = user_ns


get_ipython().register_magic_function(scope, 'cell')

