import logging
from os.path import join
from os import getenv, makedirs
from sklearn.ensemble import RandomForestClassifier
from models.models import trainAndTestModel, optimizeModelParameters

# Random Forest Classifier results path
__RANDOM_FOREST_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'random_forest')

makedirs(__RANDOM_FOREST_RESULTS_PATH, exist_ok=True)


def optimizeRandomForestClassifierParameters(X_train, y_train):
    """
    @brief Trains a Random Forest Classifier model on the training data.

    This function trains a Random Forest Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Random Forest Classifier model.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None,10, 20, 30],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    return optimizeModelParameters(RandomForestClassifier,'Random Forest Classifier', param_grid, __RANDOM_FOREST_RESULTS_PATH,X_train, y_train)


def trainAndTestRandomForestClassifier(X_train, y_train, X_test):
    """
    @brief Trains a Random Forest Classifier model on the training data.

    This function trains a Random Forest Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Random Forest Classifier and the predictions.
    """
    logging.info("Training Random Forest Classifier")

    return trainAndTestModel(RandomForestClassifier, X_train, y_train, X_test, __RANDOM_FOREST_RESULTS_PATH)
