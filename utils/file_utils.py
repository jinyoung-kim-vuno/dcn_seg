import os


def get_or_create_dir(dir_path, mode=0o777):
    """
    Check if the given directory exists and if it doesn't then create it with the given mode
    If it exists then the mode won't have any effect

    :param dir_path:
    :param mode:
    :return:            True if the directory exists when exiting the function

    """
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path, mode)
        except OSError:
            # There may have been a race condition where multiple processes tried to create the missing
            # directory; only the first will succeed, but that's okay. But if the directory still does
            # not exist (e.g., no permission to create the directory), that's a problem.
            if not os.path.exists(dir_path):
                raise


def get_file_size(file_path, dir_path=None):
    """

    :param file_path:       full path of the file to inquire
    :param dir_path:        if provided then the file_path is treated as file name and is concatenated to dir_path
    :return:            if file doesn't exist then return -1, otherwise return its size in bytes
    """
    if dir_path is not None and file_path is not None:
        file_path = os.path.join(dir_path, file_path)

    try:
        return os.path.getsize(file_path)
    except OSError:
        return -1


def file_exists(file_path, dir_path=None):
    """

    :param file_path:       full path of the file to inquire
    :param dir_path:        if provided then the file_path is treated as file name and is concatenated to dir_path
    :return:                True iff file exists
    """
    if dir_path is not None and file_path is not None:
        file_path = os.path.join(dir_path, file_path)

    return os.path.exists(file_path)


def create_log(filepath, content, is_debug):
    if is_debug == 1:
        print(content)
        with open(filepath, 'a') as f:
            f.write('\n' + content)


def create_log_exit(filepath, content):
    print(content)
    with open(filepath, 'a') as f:
        f.write('\n' + content)
    exit()
