"""
@file prepare.py
@brief This module contains functions for preparing datasets for further analysis.

Functions:
    - checkDatasetFileExists(data_directory_path: str, file_name: str, url: str) -> None:
        Downloads the dataset file if it does not exist in the data directory.
        This function is used to check if the dataset file exists in the data directory.
        If the file does not exist, an error is raised indicating that the file is missing.

    - openDatasets(data_directory_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Opens the datasets from the specified data directory. The datasets are expected to be in CSV format.
        The function checks if the directory and the required files exist. If not, it raises a FileNotFoundError.
        It returns three pandas DataFrames corresponding to the training data, test data, and product types codes.

    - showDatasetInfo(dataset: pd.DataFrame) -> None:
        Prints the shape, head, and info of the provided pandas DataFrame. This function is useful for getting a quick overview of the dataset.

    - prepareDataset(dataset: pd.DataFrame) -> None:
        Drops the columns 'id', 'description', and 'imageid' from the dataset if they exist. The changes are made in-place (EDIT THE EXISTING DATASET, DATA IS LOST FOREVER).

    - prepareDatasets(data_directory_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Opens the datasets from the specified data directory, prepares them by dropping certain columns, and returns the prepared datasets.
        It returns three pandas DataFrames corresponding to the prepared training data, test data, and product types codes.

Note:
    All functions in this module operate on pandas DataFrames and are designed to be used in a data preprocessing pipeline.
"""

from os.path import join, exists
from os import makedirs
from pandas import read_csv


def checkDatasetFileExists(data_directory_path, file_name, url):
    """
    @brief Downloads the dataset file if it does not exist in the data directory.

    This function is used to check if the dataset file exists in the data directory.
    If the file does not exist, an error is raised indicating that the file is missing.

    @param data_directory_path The path to the data directory.
    @param file_name The name of the dataset file to be downloaded.
    @param url The URL of the dataset file to be downloaded.
    @return None
    """
    if not exists(join(data_directory_path, file_name)):
        raise FileNotFoundError(f"File {file_name} not found in the data directory. Please download it from {url} and place it at {data_directory_path}")


def openDatasets(data_directory_path):
    """
    @brief Opens the datasets from the data directory.

    This function checks if the directory and the required files exist
    If not, it raises a FileNotFoundError.

    @param data_directory_path The path to the data directory.
    @return Three pandas DataFrames corresponding to the training data, test data, and product types codes.
    """
    makedirs(data_directory_path, exist_ok=True)
    datasets_paths = {
        'X_train_update.csv': 'https://challengedata.ens.fr/participants/challenges/35/download/x-train',
        'X_test_update.csv': 'https://challengedata.ens.fr/participants/challenges/35/download/x-test',
        'Y_train.csv': 'https://challengedata.ens.fr/participants/challenges/35/download/y-train'
    }
    for file_name, file_url in datasets_paths.items():
        checkDatasetFileExists(data_directory_path, file_name, file_url)

    product_data_train = read_csv(join(data_directory_path, 'X_train_update.csv'))
    product_data_test = read_csv(join(data_directory_path, 'X_test_update.csv'))
    product_types_codes = read_csv(join(data_directory_path, 'Y_train.csv'))

    return product_data_train, product_data_test, product_types_codes


def showDatasetInfo(dataset):
    """
    @brief Prints the shape, head, and info of the dataset.

    This function is useful for getting a quick overview of the dataset.

    @param dataset The pandas DataFrame to be inspected.
    @return None
    """
    print(dataset.shape)
    print(dataset.head())
    print(dataset.info())


def prepareDataset(dataset):
    """
    @brief Drops the columns 'id', 'description', and 'imageid' from the dataset (if they exist).

    The changes are made in-place.

    @param dataset The pandas DataFrame to be prepared.
    @return None
    """
    columns_to_drop = [col for col in ['id', 'description', 'imageid'] if col in dataset.columns]
    dataset.drop(columns_to_drop, axis=1, inplace=True)


def prepareDatasets(data_directory_path):
    """
    @brief Opens the datasets from the data directory and prepares them.

    This function calls openDatasets() to open the datasets and then prepareDataset()
    to drop certain columns. It returns the prepared datasets.

    @param data_directory_path The path to the data directory.
    @return Three pandas DataFrames corresponding to the prepared training data, test data, and product types codes.
    """
    product_data_train, product_data_test, product_types_codes = openDatasets(data_directory_path)

    prepareDataset(product_data_train)
    prepareDataset(product_data_test)
    prepareDataset(product_types_codes)

    return product_data_train, product_data_test, product_types_codes
