import logging
from os.path import join
from os import getenv, makedirs
from sklearn.ensemble import GradientBoostingClassifier
from models.models import trainAndTestModel, optimizeModelParameters

# Gradient Boosting Classifier results path
__GRADIENT_BOOSTING_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'gradient_boosting')

makedirs(__GRADIENT_BOOSTING_RESULTS_PATH, exist_ok=True)


def optimizeGradientBoostingClassifierParameters(X_train, Y_train):
    """
    @brief Optimizes the Gradient Boosting Classifier.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @return The optimized Gradient Boosting Classifier.
    """
    param_grid = [
        {
            'n_estimators': 300,
            'learning_rate': 0.1
        },
        {
            'n_estimators': 300,
            'learning_rate': 0.01
        },
        {
            'n_estimators': 300,
            'learning_rate': 1
        },
        {
            'n_estimators': 300,
            'learning_rate': 10
        }
    ]

    return optimizeModelParameters(GradientBoostingClassifier,'Gradient Boosting Classifier',param_grid, __GRADIENT_BOOSTING_RESULTS_PATH, X_train, Y_train)


def trainAndTestGradientBoostingClassifier(X_train, Y_train, X_test):
    """
    @brief Trains a Gradient Boosting Classifier model on the training data.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Gradient Boosting Classifier and the predictions.
    """
    logging.debug("Training Gradient Boosting Classifier")
    return trainAndTestModel(GradientBoostingClassifier, X_train, Y_train, X_test, __GRADIENT_BOOSTING_RESULTS_PATH)
