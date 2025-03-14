# copyright (c) 2025, SIMply developers
# this file is part of the SIMply package, see LICENCE.txt for licence.
import os
from typing import List


def dataDirPath():
    """ Returns the absolute path to the SIMply data directory"""
    absoluteFilePath = os.path.dirname(__file__)
    relativePath = os.path.join(os.pardir, "data")
    dirPath = os.path.abspath(os.path.join(absoluteFilePath, relativePath))
    return dirPath


def dataFilePath(sub_dirs: List[str], file_name: str):
    """ Returns the absolute path to a file in the SIMply data directory given its file name and any subdirectories
     it is in within the data directory.

    :param sub_dirs: ordered list of any subdirectories in which the file is located
    :param file_name: filename of the file
    :return: absolute path to the file
    """
    if len(sub_dirs) > 0:
        subDirsPath = ""
        for subDir in sub_dirs:
            subDirsPath = os.path.join(subDirsPath, subDir)
        return os.path.join(dataDirPath(), subDirsPath, file_name)
    return os.path.join(dataDirPath(), file_name)
