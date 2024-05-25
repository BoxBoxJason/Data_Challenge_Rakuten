import logging
from os import makedirs, getenv
from os.path import join
from sklearn.ensemble import BaggingClassifier
from models.models import trainAndTestModel, optimizeModelParameters

# Bagging Classifier results path
__BAGGING_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'bagging')

makedirs(__BAGGING_RESULTS_PATH, exist_ok=True)

def preProcessDataset(dataset):
    """
    @brief Preprocesses the dataset.

    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Bagging Classifier")
    return dataset


def optimizeBaggingClassifierParameters(X_train, y_train):
    """
    @brief Trains a Bagging Classifier model on the training data.

    This function trains a Bagging Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Bagging Classifier model.
    """
    param_grid = {
        'n_estimators': [None, 50, 100],
        'max_samples': [1.0, 0.5],
        'max_features': [1.0, 0.5],
        'bootstrap': [True, False],
        'bootstrap_features': [True, False]
    }

    processed_X_train = preProcessDataset(X_train)

    return optimizeModelParameters(BaggingClassifier, 'Bagging Classifier', param_grid, __BAGGING_RESULTS_PATH, processed_X_train, y_train)


def trainAndTestBaggingClassifier(X_train, y_train, X_test):
    """
    @brief Trains a Bagging Classifier model on the training data.

    This function trains a Bagging Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Bagging Classifier and the predictions.
    """
    logging.debug("Training Bagging Classifier")
    return trainAndTestModel(BaggingClassifier, X_train, y_train, X_test, __BAGGING_RESULTS_PATH)
