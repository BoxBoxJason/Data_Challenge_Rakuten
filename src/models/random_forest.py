import logging
from os.path import join
from os import getenv, makedirs
from sklearn.ensemble import RandomForestClassifier
from models.models import trainAndTestModel, optimizeModelParameters, drawGraphs

# Random Forest Classifier results path
__RANDOM_FOREST_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'random_forest')

makedirs(__RANDOM_FOREST_RESULTS_PATH, exist_ok=True)


def preProcessDataset(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Random Forest Classifier")
    return dataset


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

    processed_X_train = preProcessDataset(X_train)

    return optimizeModelParameters(RandomForestClassifier,'Random Forest Classifier', param_grid, __RANDOM_FOREST_RESULTS_PATH,processed_X_train, y_train)


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
    logging.debug("Training Random Forest Classifier")

    processed_X_train = preProcessDataset(X_train)
    processed_X_test = preProcessDataset(X_test)

    return trainAndTestModel(RandomForestClassifier, processed_X_train, y_train, processed_X_test, __RANDOM_FOREST_RESULTS_PATH)

def drawGraphsRandomForest():
    """
    @brief Draws graphs for the Random Forest Classifier results.

    This function draws graphs for the Random Forest Classifier results.
    Display mean test scores for each hyperparameter which are max_depth, max_features and n_estimators.
    """
    logging.debug("Drawing Random Forest Classifier graphs")
    drawGraphs('Random Forest Classifier', __RANDOM_FOREST_RESULTS_PATH, 'max_features', 'n_estimators', 'max_depth')