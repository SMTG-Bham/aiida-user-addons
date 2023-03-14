"""
Decorators for simplifying writing function dealing with AiiDA models
"""
from functools import wraps

from aiida.orm import FolderData, Node, load_node


def with_node(f):
    """Decorator to ensure that the function receives a node"""

    @wraps(f)
    def wrapped(*args, **kwargs):
        if not isinstance(args[0], Node):
            args = list(args)
            args[0] = load_node(args[0])
        return f(*args, **kwargs)

    return wrapped


def with_retrieved(f):
    """Decorator to ensure that the function receives a FolderData"""

    @wraps(f)
    @with_node
    def wrapped(*args, **kwargs):
        args = list(args)
        # Access "retrieved" unless a FolderData is passed directly
        if not isinstance(args[0], FolderData):
            args[0] = args[0].outputs.retrieved
        return f(*args, **kwargs)

    return wrapped


def with_retrieved_content(name):
    """Ensure that the first argument of the function is a line of file contents"""

    def _inner(f):
        """Inner wrapped function"""

        @wraps(f)
        @with_retrieved
        def wrapped(*args, **kwargs):
            args = list(args)
            if isinstance(args[0], FolderData):
                # Parse files if a FolderData node is passed
                with args[0].open(name) as handle:
                    content = handle.readlines()
                args[0] = content
            return f(*args, **kwargs)

        return wrapped

    return _inner


def with_retrieved_handle(name):
    """Ensure that the first argument of the function is a line of file contents"""

    def _inner(f):
        """Inner wrapped function"""

        @wraps(f)
        @with_retrieved
        def wrapped(*args, **kwargs):
            args = list(args)
            if isinstance(args[0], FolderData):
                # Parse files if a FolderData node is passed
                with args[0].open(name) as handle:
                    args[0] = handle
                    return f(*args, **kwargs)
            return f(*args, **kwargs)

        return wrapped

    return _inner


def with_node_list(f):
    """Decorator to ensure that the function receives a list of nodes in its first
    positional argument"""

    @wraps(f)
    def wrapped(*args, **kwargs):
        arg0 = list(args[0])
        for idx, item in enumerate(arg0):
            if not isinstance(item, Node):
                arg0[idx] = load_node(arg0[idx])
        args = list(args)
        args[0] = arg0
        return f(*args, **kwargs)

    return wrapped
