import logging
from os import getenv, makedirs
from os.path import join
from sklearn.svm import SVC, LinearSVC, NuSVC
from models.models import trainAndTestModel, optimizeModelParameters

# SVC Classifier results directory path
__SVC_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'svc')

# LinearSVC Classifier results directory path
__LINEAR_SVC_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'linear_svc')

# NuSVC Classifier results directory path
__NU_SVC_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'nu_svc')

makedirs(__SVC_RESULTS_PATH, exist_ok=True)
makedirs(__LINEAR_SVC_RESULTS_PATH, exist_ok=True)
makedirs(__NU_SVC_RESULTS_PATH, exist_ok=True)

def preProcessDatasetSVC(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for SVC Classifier")
    return dataset


def optimizeSVCParameters(X_train, y_train):
    """
    @brief Trains a SVC Classifier model on the training data.

    This function trains a SVC Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained SVC Classifier model.
    """
    param_grid = param_grid = [
        {'C': [0.1, 1, 10], 'kernel': ['linear']},
        {'C': [0.1, 1, 10], 'kernel': ['rbf'], 'gamma': ['scale', 'auto']},
        {'C': [0.1, 1, 10], 'kernel': ['poly'], 'degree': [2, 3], 'gamma': ['scale', 'auto']},
        {'C': [0.1, 1, 10], 'kernel': ['sigmoid'], 'gamma': ['scale', 'auto']}
    ]

    processed_X_train = preProcessDatasetSVC(X_train)

    return optimizeModelParameters(SVC,'SVC', param_grid, __SVC_RESULTS_PATH, processed_X_train,y_train)


def trainAndTestSVC(X_train, y_train, X_test):
    """
    @brief Trains a SVC Classifier model on the training data.

    This function trains a SVC Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained SVC Classifier and the predictions.
    """
    logging.debug("Training SVC model")

    processed_X_train = preProcessDatasetSVC(X_train)
    processed_X_test = preProcessDatasetSVC(X_test)

    return trainAndTestModel(SVC, processed_X_train, y_train, processed_X_test, __SVC_RESULTS_PATH)


def preProcessDatasetLinearSVC(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for LinearSVC Classifier")
    return dataset


def optimizeLinearSVCParameters(X_train, y_train):
    """
    @brief Trains a Linear SVC Classifier model on the training data.

    This function trains a Linear SVC Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Linear SVC Classifier model.
    """
    param_grid = {
        'C': [0.1, 1, 10],
        'loss': ['hinge', 'squared_hinge'],
        'dual': [True, False],
        'penalty': ['l1', 'l2'],
        'multi_class': ['ovr', 'crammer_singer'],
    }

    processed_X_train = preProcessDatasetLinearSVC(X_train)

    return optimizeModelParameters(LinearSVC,'Linear SVC Classifier', param_grid, __LINEAR_SVC_RESULTS_PATH, processed_X_train,y_train)


def trainAndTestLinearSVC(X_train, y_train, X_test):
    """
    @brief Trains a Linear SVC Classifier model on the training data.

    This function trains a Linear SVC Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Linear SVC Classifier and the predictions.
    """
    logging.debug("Training Linear SVC model")

    processed_X_train = preProcessDatasetLinearSVC(X_train)
    processed_X_test = preProcessDatasetLinearSVC(X_test)

    return trainAndTestModel(LinearSVC, processed_X_train, y_train, processed_X_test, __LINEAR_SVC_RESULTS_PATH)


def preProcessDatasetNuSVC(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for NuSVC Classifier")
    return dataset


def optimizeNuSVCParameters(X_train, y_train):
    """
    @brief Trains a NuSVC Classifier model on the training data.

    This function trains a NuSVC Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained NuSVC Classifier model.
    """
    param_grid = {
        'nu': [0.1, 0.5, 0.9],
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto'],
    }

    processed_X_train = preProcessDatasetNuSVC(X_train)

    return optimizeModelParameters(NuSVC,'NuSVC Classifier', param_grid, __NU_SVC_RESULTS_PATH, processed_X_train,y_train)


def trainAndTestNuSVC(X_train, y_train, X_test):
    """
    @brief Trains a NuSVC Classifier model on the training data.

    This function trains a NuSVC Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained NuSVC Classifier and the predictions.
    """
    logging.debug("Training NuSVC model")

    processed_X_train = preProcessDatasetNuSVC(X_train)
    processed_X_test = preProcessDatasetNuSVC(X_test)

    return trainAndTestModel(NuSVC, processed_X_train, y_train, processed_X_test, __NU_SVC_RESULTS_PATH)
