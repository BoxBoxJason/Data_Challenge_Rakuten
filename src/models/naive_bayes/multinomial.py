import logging
from os import getenv, makedirs
from os.path import join
from sklearn.naive_bayes import MultinomialNB
from models.models import trainAndTestModel, optimizeModelParameters

# Naive Bayes Multinomial directory results path
__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'multinomialNB')

makedirs(__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH, exist_ok=True)


def optimizeMultinomialNaiveBayesClassifierParameters(X_train, y_train):
    """
    @brief Trains a Multinomial Naive Bayes Classifier model on the training data.

    This function trains a Multinomial Naive Bayes Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Multinomial Naive Bayes Classifier model.
    """
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'fit_prior': [True, False]
    }

    return optimizeModelParameters(MultinomialNB,'Multinomial Naive Bayes Classifier',param_grid,__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH,X_train,y_train)


def trainAndTestMultinomialNaiveBayesClassifier(X_train, y_train, X_test):
    """
    @brief Trains a Multinomial Naive Bayes Classifier model on the training data.

    This function trains a Multinomial Naive Bayes Classifier model on the training data using the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @param best_params The best hyperparameters for the Multinomial Naive Bayes Classifier.
    @return The trained Multinomial Naive Bayes Classifier model.
    """
    logging.info("Training Multinomial Naive Bayes Classifier model")
    return trainAndTestModel(MultinomialNB, X_train, y_train, X_test, __NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH)
