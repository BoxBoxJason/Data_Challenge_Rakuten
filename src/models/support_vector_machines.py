import logging
from os import getenv, makedirs
from os.path import join
from sklearn.svm import SVC, LinearSVC, NuSVC
from models.models import trainClassifier, predictTestDataset, optimizeModelParameters

# SVC Classifier results directory path
__SVC_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'svc')
# SVC predicted csv path
__SVC_PREDICTED_CSV_PATH = join(__SVC_RESULTS_PATH, 'predicted_Y_test.csv')

# LinearSVC Classifier results directory path
__LINEAR_SVC_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'linear_svc')
# LinearSVC predicted csv path
__LINEAR_SVC_PREDICTED_CSV_PATH = join(__LINEAR_SVC_RESULTS_PATH, 'predicted_Y_test.csv')

# NuSVC Classifier results directory path
__NU_SVC_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'nu_svc')
# NuSVC predicted csv path
__NU_SVC_PREDICTED_CSV_PATH = join(__NU_SVC_RESULTS_PATH, 'predicted_Y_test.csv')

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


def trainSVC(X_train, y_train, model_params={}):
    """
    @brief Trains a SVC Classifier model on the training data.

    This function trains a SVC Classifier model on the training data.

    @param X_train The training data.
    @param y_train The target variable.
    @param model_params The parameters for the SVC Classifier model.
    @return The trained SVC Classifier model.
    """
    logging.debug("Training SVC model")

    processed_X_train = preProcessDatasetSVC(X_train)
    model = trainClassifier(SVC, model_params, processed_X_train, y_train)

    return model


def predictSVC(svc, X_test, save_results=False):
    logging.debug("Predicting target variable using SVC model")
    processed_X_test = preProcessDatasetSVC(X_test)

    if save_results:
        logging.info(f"Saving predicted target variable at {__SVC_PREDICTED_CSV_PATH}")

    predicted_Y_test = predictTestDataset(svc, processed_X_test, __SVC_PREDICTED_CSV_PATH if save_results else None)

    return predicted_Y_test


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


def trainLinearSVC(X_train, y_train, model_params={}):
    """
    @brief Trains a Linear SVC Classifier model on the training data.

    This function trains a Linear SVC Classifier model on the training data.

    @param X_train The training data.
    @param y_train The target variable.
    @param model_params The parameters for the Linear SVC Classifier model.
    @return The trained Linear SVC Classifier model.
    """
    logging.debug("Training Linear SVC model")

    processed_X_train = preProcessDatasetLinearSVC(X_train)
    model = trainClassifier(LinearSVC, model_params, processed_X_train, y_train)

    return model


def predictLinearSVC(linear_svc, X_test, save_results=False):
    logging.debug("Predicting target variable using Linear SVC model")
    processed_X_test = preProcessDatasetLinearSVC(X_test)

    if save_results:
        logging.info(f"Saving predicted target variable at {__LINEAR_SVC_PREDICTED_CSV_PATH}")

    predicted_Y_test = predictTestDataset(linear_svc, processed_X_test, __LINEAR_SVC_PREDICTED_CSV_PATH if save_results else None)

    return predicted_Y_test


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


def trainNuSVC(X_train, y_train, model_params={}):
    """
    @brief Trains a NuSVC Classifier model on the training data.

    This function trains a NuSVC Classifier model on the training data.

    @param X_train The training data.
    @param y_train The target variable.
    @param model_params The parameters for the NuSVC Classifier model.
    @return The trained NuSVC Classifier model.
    """
    logging.debug("Training NuSVC model")

    processed_X_train = preProcessDatasetNuSVC(X_train)
    model = trainClassifier(NuSVC, model_params, processed_X_train, y_train)

    return model


def predictNuSVC(nu_svc, X_test, save_results=False):
    logging.debug("Predicting target variable using Linear SVC model")
    processed_X_test = preProcessDatasetLinearSVC(X_test)

    if save_results:
        logging.info(f"Saving predicted target variable at {__NU_SVC_PREDICTED_CSV_PATH}")

    predicted_Y_test = predictTestDataset(nu_svc, processed_X_test, __NU_SVC_PREDICTED_CSV_PATH if save_results else None)

    return predicted_Y_test
