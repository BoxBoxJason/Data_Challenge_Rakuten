import logging
from os import getenv, makedirs
from os.path import join
from sklearn.naive_bayes import ComplementNB
from models.models import trainAndTestModel, optimizeModelParameters

#Naive Bayes Classifer Complement directory results path
__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'complementNB')

makedirs(__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH, exist_ok=True)


def optimizeComplementNaiveBayesClassifierParameters(X_train, y_train):
    """
    @brief Trains a Complement Naive Bayes Classifier model on the training data.

    This function trains a Complement Naive Bayes Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Complement Naive Bayes Classifier model.
    """
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'norm': [True, False]
    }

    return optimizeModelParameters(ComplementNB,'Complement Naive Bayes Classifier',param_grid,__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH,X_train,y_train)


def trainAndTestComplementNaiveBayesClassifier(X_train, y_train, X_test):
    """
    @brief Trains a Complement Naive Bayes Classifier model on the training data.

    This function trains a Complement Naive Bayes Classifier model on the training data using the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @param best_params The best hyperparameters for the Complement Naive Bayes Classifier.
    @return The trained Complement Naive Bayes Classifier model.
    """
    logging.info("Training Complement Naive Bayes Classifier model")
    return trainAndTestModel(ComplementNB, X_train, y_train, X_test, __NAIVE_BAYES_COMPLEMENT_RESULTS_PATH)
