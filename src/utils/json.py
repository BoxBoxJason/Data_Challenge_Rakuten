from json import load, dump

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
