#/usr/bin/env python3
"""Text loader for preprocessing"""


def text_loader(file_list, encoding="utf8"):
    """Loader files within generator object for low memory usage"""
    for file in file_list:
        with open(file, encoding=encoding, errors="ignore") as input_file:
            text = input_file.read()
        yield text

