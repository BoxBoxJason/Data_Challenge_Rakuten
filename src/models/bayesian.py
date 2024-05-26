import logging
from os import getenv, makedirs
from os.path import join
from sklearn.naive_bayes import ComplementNB, MultinomialNB, GaussianNB, BernoulliNB
from models.models import trainAndTestModel, optimizeModelParameters, drawGraphs

#Naive Bayes Classifer Gaussian directory results path
__NAIVE_BAYES_GAUSSIAN_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'gaussianNB')

#Naive Bayes Classifer Complement directory results path
__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'complementNB')

# Naive Bayes Multinomial directory results path
__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'multinomialNB')

# Naive Bayes Bernoulli directory results path
__NAIVE_BAYES_BERNOULLI_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'bernoulliNB')

makedirs(__NAIVE_BAYES_GAUSSIAN_RESULTS_PATH, exist_ok=True)
makedirs(__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH, exist_ok=True)
makedirs(__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH, exist_ok=True)
makedirs(__NAIVE_BAYES_BERNOULLI_RESULTS_PATH, exist_ok=True)

def preProcessDatasetGaussian(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Naive Bayes Gaussian Classifier")
    return dataset


def optimizeNaiveBayesGaussianClassifierParameters(X_train, y_train):
    """
    @brief Trains a Naive Bayes Gaussian Classifier model on the training data.

    This function trains a Naive Bayes Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Naive Bayes Classifier model.
    """
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }

    processed_X_train = preProcessDatasetGaussian(X_train)

    return optimizeModelParameters(GaussianNB,'Gaussian Naive Bayes Classifier',param_grid,__NAIVE_BAYES_GAUSSIAN_RESULTS_PATH,processed_X_train,y_train)


def trainAndTestNaiveBayesGaussianClassifier(X_train, y_train, X_test):
    """
    @brief Trains a Naive Bayes Gaussian Classifier model on the training data.

    This function trains a Naive Bayes Gaussian Classifier model on the training data using the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @param best_params The best hyperparameters for the Naive Bayes Gaussian Classifier.
    @return The trained Naive Bayes Gaussian Classifier model.
    """
    logging.debug("Training Naive Bayes Gaussian Classifier model")

    processed_X_train = preProcessDatasetGaussian(X_train)
    processed_X_test = preProcessDatasetGaussian(X_test)

    return trainAndTestModel(GaussianNB, processed_X_train, y_train, processed_X_test, __NAIVE_BAYES_GAUSSIAN_RESULTS_PATH)


def preProcessDatasetComplement(dataset):
    """
    @brief Preprocesses the dataset.

    This function preprocesses the dataset.

    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Naive Bayes Complement Classifier")
    return dataset


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

    processed_X_train = preProcessDatasetComplement(X_train)

    return optimizeModelParameters(ComplementNB,'Complement Naive Bayes Classifier',param_grid,__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH,processed_X_train,y_train)


def trainAndTestComplementNaiveBayesClassifier(X_train, y_train, X_test):
    """
    @brief Trains a Complement Naive Bayes Classifier model on the training data.

    This function trains a Complement Naive Bayes Classifier model on the training data using the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @param best_params The best hyperparameters for the Complement Naive Bayes Classifier.
    @return The trained Complement Naive Bayes Classifier model.
    """
    logging.debug("Training Complement Naive Bayes Classifier model")

    processed_X_train = preProcessDatasetComplement(X_train)
    processed_X_test = preProcessDatasetComplement(X_test)

    return trainAndTestModel(ComplementNB, processed_X_train, y_train, processed_X_test, __NAIVE_BAYES_COMPLEMENT_RESULTS_PATH)

def drawGraphsComplementNaiveBayes():
    """
    @brief Draws graphs for the Complement Naive Bayes Classifier model.
    """
    drawGraphs('Complement Naive Bayes Classifier', __NAIVE_BAYES_COMPLEMENT_RESULTS_PATH, 'norm', 'alpha')


def preProcessDatasetMultinomial(dataset):
    """
    @brief Preprocesses the dataset.

    This function preprocesses the dataset.

    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Naive Bayes Multinomial Classifier")
    return dataset


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

    processed_X_train = preProcessDatasetMultinomial(X_train)

    return optimizeModelParameters(MultinomialNB,'Multinomial Naive Bayes Classifier',param_grid,__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH,processed_X_train,y_train)


def trainAndTestMultinomialNaiveBayesClassifier(X_train, y_train, X_test):
    """
    @brief Trains a Multinomial Naive Bayes Classifier model on the training data.

    This function trains a Multinomial Naive Bayes Classifier model on the training data using the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @param best_params The best hyperparameters for the Multinomial Naive Bayes Classifier.
    @return The trained Multinomial Naive Bayes Classifier model.
    """
    logging.debug("Training Multinomial Naive Bayes Classifier model")

    processed_X_train = preProcessDatasetMultinomial(X_train)
    processed_X_test = preProcessDatasetMultinomial(X_test)

    return trainAndTestModel(MultinomialNB, processed_X_train, y_train, processed_X_test, __NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH)

def drawGraphsMultinomialNaiveBayes():
    """
    @brief Draws graphs for the Multinomial Naive Bayes Classifier model.
    """
    drawGraphs('Multinomial Naive Bayes Classifier', __NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH, 'fit_prior', 'alpha')

def preProcessDatasetBernoulli(dataset):
    """
    @brief Preprocesses the dataset.

    This function preprocesses the dataset.

    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Naive Bayes Bernoulli Classifier")
    return dataset


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

    processed_X_train = preProcessDatasetBernoulli(X_train)

    return optimizeModelParameters(BernoulliNB,'Bernoulli Naive Bayes Classifier',param_grid,__NAIVE_BAYES_BERNOULLI_RESULTS_PATH,processed_X_train,y_train)


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

    processed_X_train = preProcessDatasetBernoulli(X_train)
    processed_X_test = preProcessDatasetBernoulli(X_test)

    return trainAndTestModel(BernoulliNB, processed_X_train, y_train, processed_X_test, __NAIVE_BAYES_BERNOULLI_RESULTS_PATH)

def drawGraphsBernoulliNaiveBayes():
    """
    @brief Draws graphs for the Bernoulli Naive Bayes Classifier model.
    """
    drawGraphs('Bernoulli Naive Bayes Classifier', __NAIVE_BAYES_BERNOULLI_RESULTS_PATH, 'fit_prior', 'alpha', 'binarize')
