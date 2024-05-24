import logging
from os import getenv, makedirs
from os.path import join
from sklearn.naive_bayes import ComplementNB, MultinomialNB, GaussianNB, BernoulliNB
from models.models import predictTestDataset, trainClassifier, optimizeModelParameters

# Naive Bayes Classifier results path
__NAIVE_BAYES_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'naive_bayes_results')

#Naive Bayes Classifer Gaussian directory results path
__NAIVE_BAYES_GAUSSIAN_RESULTS_PATH = join(__NAIVE_BAYES_RESULTS_PATH, 'gaussian')
# Naive Bayes Gaussian predicted csv path
__NAIVE_BAYES_GAUSSIAN_PREDICTED_CSV_PATH = join(__NAIVE_BAYES_GAUSSIAN_RESULTS_PATH, 'predicted_Y_test.csv')

#Naive Bayes Classifer Complement directory results path
__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH = join(__NAIVE_BAYES_RESULTS_PATH, 'complement')
# Naive Bayes Complement predicted csv path
__NAIVE_BAYES_COMPLEMENT_PREDICTED_CSV_PATH = join(__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH, 'predicted_Y_test.csv')

# Naive Bayes Multinomial directory results path
__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH = join(__NAIVE_BAYES_RESULTS_PATH, 'multinomial')
# Naive Bayes Multinomial predicted csv path
__NAIVE_BAYES_MULTINOMIAL_PREDICTED_CSV_PATH = join(__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH, 'predicted_Y_test.csv')

# Naive Bayes Bernoulli directory results path
__NAIVE_BAYES_BERNOULLI_RESULTS_PATH = join(__NAIVE_BAYES_RESULTS_PATH, 'bernoulli')
# Naive Bayes Bernoulli predicted csv path
__NAIVE_BAYES_BERNOULLI_PREDICTED_CSV_PATH = join(__NAIVE_BAYES_BERNOULLI_RESULTS_PATH, 'predicted_Y_test.csv')

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


def trainNaiveBayesGaussianClassifier(X_train, y_train, model_params={}):
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
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'norm': [True, False]
    }

    processed_X_train = preProcessDatasetComplement(X_train)

    return optimizeModelParameters(ComplementNB,'Complement Naive Bayes Classifier',param_grid,__NAIVE_BAYES_COMPLEMENT_RESULTS_PATH,processed_X_train,y_train)


def trainComplementNaiveBayesClassifier(X_train, y_train, model_params={}):
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
    param_grid = {
        'alpha': [0.1, 0.5, 1.0, 1.5, 2.0],
        'fit_prior': [True, False]
    }

    processed_X_train = preProcessDatasetMultinomial(X_train)

    return optimizeModelParameters(MultinomialNB,'Multinomial Naive Bayes Classifier',param_grid,__NAIVE_BAYES_MULTINOMIAL_RESULTS_PATH,processed_X_train,y_train)


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


def trainBernoulliNaiveBayesClassifier(X_train, y_train, model_params=None):
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
    if model_params is None:
        model_params = {}
    bnb = trainClassifier(BernoulliNB, model_params, processed_X_train, y_train)

    return bnb


def predictBernoulliNaiveBayesClassifier(bnb, X_test, save_results=False):
    """
    @brief Predicts the target variable using the Bernoulli Naive Bayes Classifier model.

    This function predicts the target variable using the Bernoulli Naive Bayes Classifier model.

    @param X_test The test data.
    @param bnb The Bernoulli Naive Bayes Classifier model.
    @return The predicted target variable.
    """
    logging.debug("Predicting target variable using Bernoulli Naive Bayes Classifier model")
    processed_X_test = preProcessDatasetBernoulli(X_test)

    if save_results:
        logging.info(f"Saving predicted target variable at {__NAIVE_BAYES_BERNOULLI_PREDICTED_CSV_PATH}")

    predicted_Y_test = predictTestDataset(bnb, processed_X_test, __NAIVE_BAYES_BERNOULLI_PREDICTED_CSV_PATH if save_results else None)

    return predicted_Y_test
