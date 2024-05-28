import logging
from os import getenv, makedirs
from os.path import join
from sklearn.tree import DecisionTreeClassifier
from models.models import trainAndTestModel, optimizeModelParameters

# Decision Tree Classifier results directory path
__DECISION_TREE_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'decision_tree')

makedirs(__DECISION_TREE_RESULTS_PATH, exist_ok=True)


def optimizeDecisionTreeParameters(X_train, y_train):
    """
    @brief Trains a Decision Tree Classifier model on the training data.

    This function trains a Decision Tree Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Decision Tree Classifier model.
    """
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'splitter': ['best', 'random'],
        'max_features': ['auto', 'sqrt', 'log2'],
        'ccp_alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    }

    return optimizeModelParameters(DecisionTreeClassifier, 'Decision Tree Classifier', param_grid, __DECISION_TREE_RESULTS_PATH, X_train, y_train)


def trainAndTestDecisionTreeModel(X_train, X_test, y_train):
    """
    @brief Trains and tests a Decision Tree Classifier model.

    This function trains a Decision Tree Classifier model on the training data and
    tests it on the testing data.

    @param X_train The training data.
    @param X_test The testing data.
    @param y_train The training target variable.
    @return None
    """
    logging.debug("Training Decision Tree Classifier model")
    trainAndTestModel(DecisionTreeClassifier, X_train, y_train, X_test, __DECISION_TREE_RESULTS_PATH)
