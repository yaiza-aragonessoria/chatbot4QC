# content of test_file_manager.py

import datetime
import os
import sys

import pytest
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import interface


# Define a fixture to create a temporary folder for testing
@pytest.fixture
def temp_folder(tmpdir):
    return str(tmpdir.mkdir("test_folder"))


# Test the FileManager class
def test_filemanager_get_latest_file_no_files(temp_folder):
    file_manager = interface.FileManager(temp_folder)
    assert file_manager.get_latest_file() is None


def test_filemanager_get_latest_file_with_files(temp_folder):
    file_manager = interface.FileManager(temp_folder)

    # Create some test files in the temporary folder
    file_names = ['file1.txt', 'file2.txt', 'file3.txt']
    for file_name in file_names:
        with open(os.path.join(temp_folder, file_name), 'w') as file:
            file.write("Test data")
    file_manager.get_latest_file()

    assert file_manager.file_name == 'file3.txt'


def test_filemanager_get_latest_file_with_newest_g_file(temp_folder):
    file_manager = interface.FileManager(temp_folder)

    # Create some test files in the temporary folder
    file_names = ['file1.txt', 'file2.txt', 'g_file.txt']
    for file_name in file_names:
        with open(os.path.join(temp_folder, file_name), 'w') as file:
            file.write("Test data")

    file_manager.get_latest_file()

    # The latest file should be 'questions{current_date}.txt'
    assert file_manager.file_name == f"questions{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.txt"


def test_filemanager_file_name_after_get_latest_file(temp_folder):
    file_manager = interface.FileManager(temp_folder)

    # Create some test files in the temporary folder
    file_names = ['file1.txt', 'file2.txt', 'g_file.txt']
    for file_name in file_names:
        with open(os.path.join(temp_folder, file_name), 'w') as file:
            file.write("Test data")

    # After calling get_latest_file, the file_name attribute should be set
    file_manager.get_latest_file()
    print("")
    print(file_manager.file_name)
    print("")
    assert file_manager.file_name is not None
    assert file_manager.file_name.startswith("questions")
    assert file_manager.file_name.endswith(".txt")