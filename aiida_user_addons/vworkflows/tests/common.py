"""
Some common module
"""
import os


def data_path(*args):
    return os.path.join(__file__, '../test_data', *args)
