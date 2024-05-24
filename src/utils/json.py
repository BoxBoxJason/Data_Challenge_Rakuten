from json import load, dump
from numpy import ndarray

def loadJson(file_path):
    """
    @brief Load a JSON file from the specified path.

    This function is used to load a JSON file from the specified path.

    @param file_path The path to the JSON file.
    @return The JSON data.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = load(file)
    return data


def saveJson(data, file_path):
    """
    @brief Save JSON data to the specified file path.

    This function is used to save JSON data to the specified file path.

    @param data The JSON data to be saved.
    @param file_path The path to save the JSON file.
    @return None
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        dump(data, file, indent=4)


def convertToSerializable(obj):
    """
    Convert a dictionary with non-serializable numpy arrays to a serializable format.

    Args:
        obj (dict): The dictionary to convert.

    Returns:
        dict: The converted dictionary with all non-serializable types converted to serializable ones.
    """
    if isinstance(obj, ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convertToSerializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convertToSerializable(i) for i in obj]
    else:
        return obj
