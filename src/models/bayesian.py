import logging
from os import getenv, makedirs
from os.path import join
from sklearn.naive_bayes import ComplementNB, MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV
from utils.json import saveJson, convertToSerializable
from models.models import predictTestDataset, trainClassifier

# Naive Bayes Classifier results path
__NAIVE_BAYES_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'naive_bayes_results')
#Naive Bayes Classifer Gaussian directory results path
__NAIVE_BAYES_GAUSSIAN_RESULTS_PATH = join(__NAIVE_BAYES_RESULTS_PATH, 'gaussian')
# Best parameters for Naive Bayes Gaussian Classifier path
__NAIVE_BAYES_GAUSSIAN_BEST_PARAMS_PATH = join(__NAIVE_BAYES_GAUSSIAN_RESULTS_PATH, 'best_params.json')
# All tests results for Naive Bayes Gaussian Classifier path
__NAIVE_BAYES_GAUSSIAN_ALL_TESTS_RESULTS_PATH = join(__NAIVE_BAYES_GAUSSIAN_RESULTS_PATH, 'all_tests_results.json')
# Naive Bayes Gaussian predicted csv path
__NAIVE_BAYES_GAUSSIAN_PREDICTED_CSV_PATH = join(__NAIVE_BAYES_GAUSSIAN_RESULTS_PATH, 'predicted_Y_test.csv')

#Naive Bayes Classifer Complement directory results path
__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH = join(__NAIVE_BAYES_RESULTS_PATH, 'complement')
# Best parameters for Naive Bayes Complement Classifier path
__NAIVE_BAYES_COMPLEMENT_BEST_PARAMS_PATH = join(__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH, 'best_params.json')
# All tests results for Naive Bayes Complement Classifier path
__NAIVE_BAYES_COMPLEMENT_ALL_TESTS_RESULTS_PATH = join(__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH, 'all_tests_results.json')
# Naive Bayes Complement predicted csv path
__NAIVE_BAYES_COMPLEMENT_PREDICTED_CSV_PATH = join(__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH, 'predicted_Y_test.csv')

# Naive Bayes Multinomial directory results path
__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH = join(__NAIVE_BAYES_RESULTS_PATH, 'multinomial')
# Best parameters for Naive Bayes Multinomial Classifier path
__NAIVE_BAYES_MULTINOMIAL_BEST_PARAMS_PATH = join(__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH, 'best_params.json')
# All tests results for Naive Bayes Multinomial Classifier path
__NAIVE_BAYES_MULTINOMIAL_ALL_TESTS_RESULTS_PATH = join(__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH, 'all_tests_results.json')
# Naive Bayes Multinomial predicted csv path
__NAIVE_BAYES_MULTINOMIAL_PREDICTED_CSV_PATH = join(__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH, 'predicted_Y_test.csv')

makedirs(__NAIVE_BAYES_RESULTS_PATH, exist_ok=True)
makedirs(__NAIVE_BAYES_GAUSSIAN_RESULTS_PATH, exist_ok=True)
makedirs(__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH, exist_ok=True)
makedirs(__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH, exist_ok=True)

def preProcessDatasetGaussian(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Naive Bayes Gaussian Classifier")
    return dataset


def optimizeNaiveBayesClassifierParameters(X_train, y_train):
    """
    @brief Trains a Naive Bayes Classifier model on the training data.

    This function trains a Naive Bayes Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Naive Bayes Classifier model.
    """
    logging.debug("Training Naive Bayes Classifier model")
    nb = GaussianNB()
    param_grid = {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
    }

    grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1_weighted')

    processed_X_train = preProcessDatasetGaussian(X_train)
    grid_search.fit(processed_X_train, y_train)

    logging.info(f"Best parameters for Naive Bayes Classifier: {grid_search.best_params_}")
    logging.info(f"Best score for Naive Bayes Classifier: {grid_search.best_score_}")

    logging.info(f"Saving best parameters at {__NAIVE_BAYES_GAUSSIAN_BEST_PARAMS_PATH}")
    saveJson(grid_search.best_params_, __NAIVE_BAYES_GAUSSIAN_BEST_PARAMS_PATH)

    logging.info(f"Saving all tests results at {__NAIVE_BAYES_GAUSSIAN_ALL_TESTS_RESULTS_PATH}")
    saveJson(convertToSerializable(grid_search.cv_results_), __NAIVE_BAYES_GAUSSIAN_ALL_TESTS_RESULTS_PATH)

    return grid_search.best_params_


def trainNaiveBayesGaussianClassifier(X_train, y_train, model_params=None):
    """
    @brief Trains a Naive Bayes Classifier model on the training data.

    This function trains a Naive Bayes Classifier model on the training data using the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @param best_params The best hyperparameters for the Naive Bayes Classifier.
    @return The trained Naive Bayes Gaussian Classifier model.
    """
    logging.debug("Training Naive Bayes Gaussian Classifier model")

    processed_X_train = preProcessDatasetGaussian(X_train)
    if model_params is None:
        model_params = {}
    nb = trainClassifier(GaussianNB, model_params, processed_X_train, y_train)

    return nb


def predictNaiveBayesGaussianClassifier(nb, X_test, save_results=False):
    """
    @brief Predicts the target variable using the Naive Bayes Classifier model.

    This function predicts the target variable using the Naive Bayes Classifier model.

    @param X_test The test data.
    @param nb The Naive Bayes Classifier model.
    @return The predicted target variable.
    """
    logging.debug("Predicting target variable using Naive Bayes Gaussian Classifier model")
    processed_X_test = preProcessDatasetGaussian(X_test)

    if save_results:
        logging.info(f"Saving predicted target variable at {__NAIVE_BAYES_GAUSSIAN_PREDICTED_CSV_PATH}")

    predicted_Y_test = predictTestDataset(nb, processed_X_test, __NAIVE_BAYES_GAUSSIAN_PREDICTED_CSV_PATH if save_results else None)

    return predicted_Y_test


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
    logging.debug("Training Complement Naive Bayes Classifier model")
    cnb = ComplementNB()
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'norm': [True, False]
    }

    grid_search = GridSearchCV(estimator=cnb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1_weighted')

    processed_X_train = preProcessDatasetComplement(X_train)
    grid_search.fit(processed_X_train, y_train)

    logging.info(f"Best parameters for Complement Naive Bayes Classifier: {grid_search.best_params_}")
    logging.info(f"Best score for Complement Naive Bayes Classifier: {grid_search.best_score_}")

    logging.info(f"Saving best parameters at {__NAIVE_BAYES_COMPLEMENT_BEST_PARAMS_PATH}")
    saveJson(grid_search.best_params_, __NAIVE_BAYES_COMPLEMENT_BEST_PARAMS_PATH)

    logging.info(f"Saving all tests results at {__NAIVE_BAYES_COMPLEMENT_ALL_TESTS_RESULTS_PATH}")
    saveJson(convertToSerializable(grid_search.cv_results_), __NAIVE_BAYES_COMPLEMENT_ALL_TESTS_RESULTS_PATH)

    return grid_search.best_params_


def trainComplementNaiveBayesClassifier(X_train, y_train, model_params=None):
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
    if model_params is None:
        model_params = {}
    cnb = trainClassifier(ComplementNB, model_params, processed_X_train, y_train)

    return cnb


def predictComplementNaiveBayesClassifier(cnb, X_test, save_results=False):
    """
    @brief Predicts the target variable using the Complement Naive Bayes Classifier model.

    This function predicts the target variable using the Complement Naive Bayes Classifier model.

    @param X_test The test data.
    @param cnb The Complement Naive Bayes Classifier model.
    @return The predicted target variable.
    """
    logging.debug("Predicting target variable using Complement Naive Bayes Classifier model")
    processed_X_test = preProcessDatasetComplement(X_test)

    if save_results:
        logging.info(f"Saving predicted target variable at {__NAIVE_BAYES_COMPLEMENT_PREDICTED_CSV_PATH}")

    predicted_Y_test = predictTestDataset(cnb, processed_X_test, __NAIVE_BAYES_COMPLEMENT_PREDICTED_CSV_PATH if save_results else None)

    return predicted_Y_test


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
    logging.debug("Training Multinomial Naive Bayes Classifier model")
    mnb = MultinomialNB()
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'fit_prior': [True, False]
    }

    grid_search = GridSearchCV(estimator=mnb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1_weighted')

    processed_X_train = preProcessDatasetMultinomial(X_train)
    grid_search.fit(processed_X_train, y_train)

    logging.info(f"Best parameters for Multinomial Naive Bayes Classifier: {grid_search.best_params_}")
    logging.info(f"Best score for Multinomial Naive Bayes Classifier: {grid_search.best_score_}")

    logging.info(f"Saving best parameters at {__NAIVE_BAYES_MULTINOMIAL_BEST_PARAMS_PATH}")
    saveJson(grid_search.best_params_, __NAIVE_BAYES_MULTINOMIAL_BEST_PARAMS_PATH)

    logging.info(f"Saving all tests results at {__NAIVE_BAYES_MULTINOMIAL_ALL_TESTS_RESULTS_PATH}")
    saveJson(convertToSerializable(grid_search.cv_results_), __NAIVE_BAYES_MULTINOMIAL_ALL_TESTS_RESULTS_PATH)

    return grid_search.best_params_


def trainMultinomialNaiveBayesClassifier(X_train, y_train, model_params=None):
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
    if model_params is None:
        model_params = {}
    mnb = trainClassifier(MultinomialNB, model_params, processed_X_train, y_train)

    return mnb


def predictMultinomialNaiveBayesClassifier(mnb, X_test, save_results=False):
    """
    @brief Predicts the target variable using the Multinomial Naive Bayes Classifier model.

    This function predicts the target variable using the Multinomial Naive Bayes Classifier model.

    @param X_test The test data.
    @param mnb The Multinomial Naive Bayes Classifier model.
    @return The predicted target variable.
    """
    logging.debug("Predicting target variable using Multinomial Naive Bayes Classifier model")
    processed_X_test = preProcessDatasetMultinomial(X_test)

    if save_results:
        logging.info(f"Saving predicted target variable at {__NAIVE_BAYES_MULTINOMIAL_PREDICTED_CSV_PATH}")

    predicted_Y_test = predictTestDataset(mnb, processed_X_test, __NAIVE_BAYES_MULTINOMIAL_PREDICTED_CSV_PATH if save_results else None)

    return predicted_Y_test
