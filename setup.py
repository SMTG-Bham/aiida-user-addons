#!/usr/bin/env python

import json

from setuptools import find_packages, setup

if __name__ == "__main__":
    # Provide static information in setup.json
    # such that it can be discovered automatically
    with open("setup.json") as info:
        kwargs = json.load(info)
    setup(packages=find_packages(), **kwargs)
