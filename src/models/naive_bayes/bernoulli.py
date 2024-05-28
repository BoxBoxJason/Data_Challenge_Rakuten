import logging
from os import getenv, makedirs
from os.path import join
from sklearn.naive_bayes import BernoulliNB
from models.models import trainAndTestModel, optimizeModelParameters

# Naive Bayes Bernoulli directory results path
__NAIVE_BAYES_BERNOULLI_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'bernoulliNB')

makedirs(__NAIVE_BAYES_BERNOULLI_RESULTS_PATH, exist_ok=True)


def optimizeBernoulliNaiveBayesClassifierParameters(X_train, y_train):
    """
    @brief Trains a Bernoulli Naive Bayes Classifier model on the training data.

    This function trains a Bernoulli Naive Bayes Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Bernoulli Naive Bayes Classifier model.
    """
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'binarize': [0.0, 0.5, 1.0, 1.5, 2.0],
        'fit_prior': [True, False]
    }

    return optimizeModelParameters(BernoulliNB,'Bernoulli Naive Bayes Classifier',param_grid,__NAIVE_BAYES_BERNOULLI_RESULTS_PATH,X_train,y_train)


def trainAndTestBernoulliNaiveBayesClassifier(X_train, y_train, X_test):
    """
    @brief Trains a Bernoulli Naive Bayes Classifier model on the training data.

    This function trains a Bernoulli Naive Bayes Classifier model on the training data using the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @param best_params The best hyperparameters for the Bernoulli Naive Bayes Classifier.
    @return The trained Bernoulli Naive Bayes Classifier model.
    """
    logging.debug("Training Bernoulli Naive Bayes Classifier model")
    return trainAndTestModel(BernoulliNB, X_train, y_train, X_test, __NAIVE_BAYES_BERNOULLI_RESULTS_PATH)
