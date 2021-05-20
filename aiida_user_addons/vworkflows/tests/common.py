"""
Some common module
"""
import os


def data_path(*args):
    path = os.path.join(os.path.split(__file__)[0], '../test_data', *args)
    return os.path.abspath(path)
