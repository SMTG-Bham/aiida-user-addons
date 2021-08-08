"""
Convenient routine for handling repository related operations
"""
from contextlib import contextmanager
import tempfile
from pathlib import Path
import shutil
import os
import re
import gzip

import aiida.orm as orm
from aiida.repository import FileType


class RepositoryMapper:
    """
    Temporary mapping repository stored in AiiDA to a folder on disk

    Since AiiDA 2.0, the repository is no longer a folder on the file system.
    This class provides an easy interface for temporarily mappng the repository onto a
    folder of the file system for easy parsing/exporting.

    While all AiiDA parser should be able to accept file objects, many of the third party tools does not.
    """

    def __init__(self, node: orm.FolderData, decompress=True):
        """Instantiate the mapper"""
        self.decompress = decompress
        self.node = node

    def to_folder(self, target_path: Path, exclude=None):
        """Write the content of a folder data onto the disk"""
        for ifile in self.node.list_objects():
            name, _ = ifile
            copy_from_aiida(name, self.node, target_path, self.decompress, exclude=exclude)

    @contextmanager
    def temporary_folder(self, exclude=None):
        """
        Get a transient folder in the temporary directory

        Args:
            exclude (str): Regular expression for the file to be ignored.

        Returns: A Path of the transient folder
        """

        target_path = tempfile.mkdtemp()
        self.to_folder(target_path, exclude=exclude)
        # Yield the path object
        yield Path(target_path)
        shutil.rmtree(target_path)


def copy_from_aiida(name: str, node, dst: Path, decompress=False, exclude=None):
    """
    Copy objects from aiida repository.

    Args:
        name (str): The full name (including the parent path) of the object.
        node (orm.Node): Node object for which the files in the repo to be copied.
        dst (Path): Path of the destination folder.

    This is a recursive function so directory copying also works.
    """

    # For check the regex the first thing because this function will be called recursively
    if exclude and re.match(exclude, name):
        return

    obj = node.get_object(name)

    # If it is a directory, copy the contents one by one
    if obj.file_type == FileType.DIRECTORY:
        for sub_obj in node.list_objects(name):
            copy_from_aiida(os.path.join(name, sub_obj.name), node, dst, exclude=exclude)
    else:
        # It is a file
        with node.open(name, mode='rb') as fsource:
            # Make parent directory if needed
            frepo_path = dst / name
            Path(frepo_path.parent).mkdir(exist_ok=True, parents=True)
            # Write the file
            if name.endswith('.gz') and decompress:
                out_path = str(frepo_path)[:-3]
                out_decompress = True
            else:
                out_decompress = False
                out_path = str(frepo_path)

            if not out_decompress:
                with open(out_path, 'wb') as fdst:
                    shutil.copyfileobj(fsource, fdst)
            else:
                gobj = gzip.GzipFile(fileobj=fsource, mode='rb')
                with open(out_path, 'wb') as fdst:
                    shutil.copyfileobj(gobj, fdst)


def save_all_repository_objects(node: orm.Node, target_path: Path, decompress=False, exclude=None):
    """Copy all objects of a node saved in the repository to the disc"""
    for ifile in node.list_objects():
        name, _ = ifile
        copy_from_aiida(name, node, target_path, decompress, exclude=exclude)
