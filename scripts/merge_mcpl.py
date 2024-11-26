#!/usr/bin/env python3
import os
import argparse
import ctypes
from Cinema.Interface import *

# Import the C function
_pt_merge_mcpl = importFunc('pt_merge_mcpl', type_voidp, [ctypes.c_char_p, ctypes.c_uint, ctypes.POINTER(ctypes.c_char_p)])

def merge(name, outputname):
    # Get current working directory
    current_directory = os.getcwd()

    # List all files in the current directory and filter them based on the given substring
    file_list = [f for f in os.listdir(current_directory) if os.path.isfile(f) and name in f]

    # Create an array of C strings (array of c_char_p)
    c_file_list = (ctypes.c_char_p * len(file_list))()

    # Fill the array with C strings
    for i, file in enumerate(file_list):
        c_file_list[i] = ctypes.c_char_p(file.encode('utf-8'))  # Store each file name as c_char_p

    # Call the C function
    _pt_merge_mcpl(outputname.encode('utf-8'), len(file_list), c_file_list)

    # Delete the files after merging
    for file in file_list:
        try:
            os.remove(file)  # Delete the file
            print(f"Merging completed. Successfully deleted: {file}")
        except FileNotFoundError:
            print(f"File not found: {file}")
        except PermissionError:
            print(f"No permission to delete file: {file}")
        except Exception as e:
            print(f"Error deleting file {file}: {e}")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Merge MCPL files and delete them after processing.")
    parser.add_argument('name', type=str, help='The name pattern to search for in the filenames.')
    parser.add_argument('-o', '--output', type=str, help='Output filename pattern to search for (optional).')

    # Parse arguments
    args = parser.parse_args()

    # Call the merge function with the determined name
    merge(args.name, args.output if args.output else args.name)

if __name__ == "__main__":
    main()