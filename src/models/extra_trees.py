import logging
from os import makedirs, getenv
from os.path import join
from sklearn.ensemble import ExtraTreesClassifier
from models.models import trainAndTestModel, optimizeModelParameters

# Extra Trees Classifier results path
__EXTRA_TREES_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'extra_trees')

makedirs(__EXTRA_TREES_RESULTS_PATH, exist_ok=True)

def preProcessDataset(dataset):
    """
    @brief Preprocesses the dataset.

    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Extra Trees Classifier")
    return dataset


def optimizeExtraTreesClassifierParameters(X_train, y_train):
    """
    @brief Trains an Extra Trees Classifier model on the training data.

    This function trains an Extra Trees Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Extra Trees Classifier model.
    """
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    processed_X_train = preProcessDataset(X_train)

    return optimizeModelParameters(ExtraTreesClassifier, 'Extra Trees Classifier', param_grid, __EXTRA_TREES_RESULTS_PATH, processed_X_train, y_train)


def trainAndTestExtraTreesClassifier(X_train, y_train, X_test):
    """
    @brief Trains an Extra Trees Classifier model on the training data.

    This function trains an Extra Trees Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @param model_params The model parameters.
    @return The trained Extra Trees Classifier and the predictions.
    """
    logging.debug("Training Extra Trees Classifier")
    return trainAndTestModel(ExtraTreesClassifier, X_train, y_train, X_test)
