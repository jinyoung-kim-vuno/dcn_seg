import os.path

__all__ = ['split_full_ext', 'strip_full_ext', 'basename_no_ext']


def split_full_ext(path):
    """
    Split a path into two pieces, similar to os.path.splitext(), except that
    they longest possible extension (containing one or more periods) is split
    off, rather than just the last extension.
    :param path: the path to split
    :type path: str|unicode
    :return: a tuple containing the path up to but not including the extension, and the extension
    """
    d = os.path.dirname(path)
    b = os.path.basename(path)
    pos = b.find('.')
    if pos <= 0:
        return path, ''
    return os.path.join(d, b[:pos]), b[pos:]


def strip_full_ext(path):
    """
    Remove a full extension (including one or more periods) from a path.

    :param path: the path to strip
    :type path: str|unicode
    :return: the portion of the path that excludes the file extension
    """
    return split_full_ext(path)[0]


def basename_no_ext(path):
    """
    Extract the basename from a path without any extension
    :param path: the path from which to extract the basename without extension
    :type path: str|unicode
    :return: the basename without extension
    """
    return split_full_ext(os.path.basename(path))[0]
