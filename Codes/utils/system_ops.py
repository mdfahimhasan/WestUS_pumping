# Author : Md Fahim Hasan
# PhD Candidate
# Colorado State university
# Fahim.Hasan@colostate.edu

import os
import shutil
import platform
from glob import glob


def makedirs(directory_list):
    """
    Make directory (if not exists) from a list of directory. Can create multiple directories if provided in the arg as
    a list.

    :param directory_list: A list of directories to create.

    :return: None.
    """
    for directory in directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)


def clean_and_make_directory(dir_path):
    """
    Removes an existing directory and all it's content. Then, makes a new directory.
    Works for a single directory.

    :param dir_path: Path of the directory.

    :return: None.
    """
    # cleaning the existing directory
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    # making a new directory
    os.makedirs(dir_path, exist_ok=True)


def copy_file(input_dir_or_file, copy_dir, search_by='*.tif', rename=None):
    """
    Copy a file to the specified directory.

    :param input_dir_or_file: File path of input directory/ Path of the file to copy.
    :param copy_dir: File path of copy directory.
    :param search_by: Default set to '*.tif'.
    :param rename: New name of file if required. Default set to None.

    :return: File path of copied file.
    """
    makedirs([copy_dir])
    if '.tif' not in input_dir_or_file:
        input_file = glob(os.path.join(input_dir_or_file, search_by))[0]
        if rename is not None:
            copied_file = os.path.join(copy_dir, f'{rename}.tif')
        else:
            file_name = os.path.basename(input_file)
            copied_file = os.path.join(copy_dir, file_name)

        shutil.copyfile(input_file, copied_file)

    else:
        if rename is not None:
            copied_file = os.path.join(copy_dir, f'{rename}.tif')
        else:
            file_name = os.path.basename(input_dir_or_file)
            copied_file = os.path.join(copy_dir, file_name)

        shutil.copyfile(input_dir_or_file, copied_file)

    return copied_file


def make_gdal_sys_call(gdal_command, args, verbose=True):
    """
    Make GDAL system call string.
    ** followed by code from Sayantan Majumdar.

    :param gdal_command: GDAL command string formatted as 'gdal_rasterize'.
    :param args: List of GDAL command.
    :param verbose: Set True to print system call info.

    :return: GDAL system call string.
    """
    if os.name == 'nt':
        # update it based on the pc's QGIS version and folderpath
        gdal_path = 'C:/Program Files/QGIS 3.22.7/OSGeo4W.bat'

        sys_call = [gdal_path] + [gdal_command] + args
        if verbose:
            print(sys_call)
        return sys_call

    else:
        print('gdal sys call not optimized for linux yet')


def assign_cpu_nodes(flags):
    """
    Dynamically assigns CPU nodes based on the operating system and the status of processing flags.

    :param flags: list of bool. Each flag indicates whether a processing step should be skipped (True) or run (False).

    :return: int or None. Number of CPU nodes assigned if processing is required; otherwise, None.
    """
    if not isinstance(flags, (list, tuple)):
        raise TypeError("Flags must be a list or tuple of boolean values.")

    # checking if any flag is False
    if any(not flag for flag in flags):

        # detecting OS
        os_name = platform.system()

        # assigning CPU nodes dynamically based on OS
        if os_name == 'Windows':  # Windows
            use_cpu_nodes = 10    # capped at 10 as Windows is local PC
        elif os_name == 'Linux':  # Linux
            use_cpu_nodes = 30    # capped at 30 as Linux is HPC
        else:
            raise ValueError(f'Unsupported OS: {os_name}. Must be Windows or Linux.')

        print(f'\n Using {use_cpu_nodes} CPU nodes on {os_name} \n...')
        return use_cpu_nodes

    else:
        return None
