import logging
from os import makedirs, getenv
from os.path import join
from sklearn.ensemble import AdaBoostClassifier
from models.models import trainAndTestModel, optimizeModelParameters, drawGraphs

# AdaBoost Classifier results path
__ADABOOST_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'adaboost')

makedirs(__ADABOOST_RESULTS_PATH, exist_ok=True)

def preProcessDataset(dataset):
    """
    @brief Preprocesses the dataset.

    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for AdaBoost Classifier")
    return dataset


def optimizeAdaBoostClassifierParameters(X_train, y_train):
    """
    @brief Trains a AdaBoost Classifier model on the training data.

    This function trains a AdaBoost Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained AdaBoost Classifier model.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [1.0, 0.1, 0.01],
        'algorithm': ['SAMME', 'SAMME.R']
    }

    processed_X_train = preProcessDataset(X_train)

    return optimizeModelParameters(AdaBoostClassifier, 'AdaBoost Classifier', param_grid, __ADABOOST_RESULTS_PATH, processed_X_train, y_train)


def trainAndTestAdaBoostClassifier(X_train, y_train, X_test):
    """
    @brief Trains a AdaBoost Classifier model on the training data.

    This function trains a AdaBoost Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained AdaBoost Classifier and the predictions.
    """
    logging.debug("Training AdaBoost Classifier")
    return trainAndTestModel(AdaBoostClassifier, X_train, y_train, X_test, __ADABOOST_RESULTS_PATH)

def drawGraphsAdaBoost():
    """
    @brief Draws graphs for AdaBoost Classifier.

    This function draws graphs for AdaBoost Classifier.
    """
    drawGraphs('AdaBoost Classifier', __ADABOOST_RESULTS_PATH, 'algorithm', 'n_estimators', 'learning_rate')