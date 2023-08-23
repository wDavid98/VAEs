"""This file contains common functions used across different parts of the code to use within models.
Made by: Edgar RP (JefeLitman)
Version: 1.2.0
"""

import os
import numpy as np

def list_directory(path):
    """Function that will list the file and folder elements of the path sorted from lower to higher in the given path.
    Args:
        path (String): The root path in where the elements will be listed.
    """
    return sorted(os.listdir(path))

def format_index(index, max_digits = 6):
    """This function format the index integer into a string with the maximun quantity of digits given.
    Args:
        index (Int): Integer to be formatted.
        max_digits (Int): How many digits must the number contain, e.g: if 4 then the range is from 0000 to 9999.
    """
    value = str(index)
    while len(value) < max_digits:
        value = '0' + value
    return value

def get_next_last_item(base_path):
    """Function than will return an integer with an id of the following element for the last item listed in the base path. Also, this method checks it the elements in the folder are formatted with the following order (xxxx_*) where xxxx is a str sequence of integers.
    Args:
        base_path (String): The root path in where the elements will be listed.
    """
    elements = [int(i.split("_")[0]) for i in list_directory(base_path)]
    if len(elements) == 0:
        return 1
    else:
        return elements[-1] + 1

def repeat_vector_to_size(array, final_size, seed):
    """This function will repeat the values of the array until it have the desired size
    Args:
        array (np.array): 1D array to be repeated in values until it get the final_size.
        final_size (Integer): Integer indicating how many values will have at the end.
        seed (Integer): Integer to be used as seed to enable the replicability of the randomization permutation
    """
    np.random.seed(seed)
    vector = np.random.permutation(array)
    original_shape = vector.shape[0]
    max_repetitions = np.ceil(final_size / original_shape).astype(np.int64)
    min_repetitions = final_size // original_shape
    final_elements = final_size % original_shape
    repeated_vector = np.concatenate([vector]*max_repetitions)[:min_repetitions*original_shape + final_elements]
    return repeated_vector
    