import logging


def getSimilarityPercentage(file1_path,file2_path):
    """
    @brief Get the similarity percentage between two files.

    This function calculates the similarity percentage between two files.

    @param file1_path The path to the first file.
    @param file2_path The path to the second file.
    @return The similarity percentage.
    """
    similarity = 0
    try:
        with open(file1_path, 'r') as file1:
            file1_content = file1.readlines()
        with open(file2_path, 'r') as file2:
            file2_content = file2.readlines()

        for i in range(min(len(file1_content), len(file2_content))):
            similarity += file1_content[i] == file2_content[i]

        if max(len(file1_content), len(file2_content)):
            similarity = similarity / max(len(file1_content), len(file2_content))
    except FileNotFoundError:
        logging.error(f"File not found at {file1_path} or {file2_path}")
    return similarity
