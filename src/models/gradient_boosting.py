import logging
from os.path import join
from os import getenv, makedirs
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from models.models import predictTestDataset, trainClassifier, optimizeModelParameters

# Gradient Boosting Classifier results path
__GRADIENT_BOOSTING_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'gradient_boosting_results')
# Gradient Boosting predicted csv path
__GRADIENT_BOOSTING_PREDICTED_CSV_PATH = join(__GRADIENT_BOOSTING_RESULTS_PATH, 'predicted_Y_test.csv')

#Hist Gradient Boosting Classifier results path
__HIST_GRADIENT_BOOSTING_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'hist_gradient_boosting_results')
# Hist Gradient Boosting predicted csv path
__HIST_GRADIENT_BOOSTING_PREDICTED_CSV_PATH = join(__HIST_GRADIENT_BOOSTING_RESULTS_PATH, 'predicted_Y_test.csv')

makedirs(__GRADIENT_BOOSTING_RESULTS_PATH, exist_ok=True)
makedirs(__HIST_GRADIENT_BOOSTING_RESULTS_PATH, exist_ok=True)


def preProcessDatasetGradientBoosting(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Gradient Boosting Classifier")
    return dataset


def optimizeGradientBoostingClassifierParameters(X_train, Y_train):
    """
    @brief Optimizes the Gradient Boosting Classifier.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @return The optimized Gradient Boosting Classifier.
    """
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 3, 4]
    }

    processed_X_train = preProcessDatasetGradientBoosting(X_train)

    return optimizeModelParameters(GradientBoostingClassifier,'Gradient Boosting Classifier',param_grid, __GRADIENT_BOOSTING_RESULTS_PATH, processed_X_train, Y_train)


def trainGradientBoostingClassifier(X_train, Y_train):
    """
    @brief Trains the Gradient Boosting Classifier.

    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @return The trained Gradient Boosting Classifier.
    """
    logging.debug("Training Gradient Boosting Classifier")

    processed_X_train = preProcessDatasetGradientBoosting(X_train)

    return trainClassifier(GradientBoostingClassifier, processed_X_train, Y_train)


def predictTestDatasetGradientBoosting(classifier, X_test,save_results=False):
    """
    @brief Predicts the test dataset with the Gradient Boosting Classifier.


    @param classifier The trained Gradient Boosting Classifier.
    @param X_test The test dataset.
    @return The predicted values for the test dataset.
    """
    logging.debug("Predicting test dataset with Gradient Boosting Classifier")

    processed_X_test = preProcessDatasetGradientBoosting(X_test)
    if save_results:
        logging.info(f"Saving predicted target variable at {__GRADIENT_BOOSTING_PREDICTED_CSV_PATH}")

    predictions = predictTestDataset(classifier, X_test, __GRADIENT_BOOSTING_PREDICTED_CSV_PATH if save_results else None)
    return predictions


def preProcessDatasetHistGradientBoosting(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Hist Gradient Boosting Classifier")
    return dataset



def optimizeHistGradientBoostingClassifierParameters(X_train, Y_train):
    """
    @brief Optimizes the Hist Gradient Boosting Classifier.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @return The optimized Hist Gradient Boosting Classifier.
    """
    param_grid = {
        'max_iter': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 1],
        'max_depth': [3, 5, 7],
        'min_samples_leaf': [10, 20, 30]
    }
    processed_X_train = preProcessDatasetHistGradientBoosting(X_train)

    return optimizeModelParameters(HistGradientBoostingClassifier,'Hist Gradient Boosting Classifier',param_grid, __HIST_GRADIENT_BOOSTING_RESULTS_PATH, processed_X_train, Y_train)


def trainHistGradientBoostingClassifier(X_train, Y_train):
    """
    @brief Trains the Hist Gradient Boosting Classifier.


    @param X_train The training dataset.
    @param Y_train The labels for the training dataset.
    @return The trained Hist Gradient Boosting Classifier.
    """
    logging.debug("Training Hist Gradient Boosting Classifier")

    processed_X_train = preProcessDatasetHistGradientBoosting(X_train)

    return trainClassifier(HistGradientBoostingClassifier, processed_X_train, Y_train)


def predictTestDatasetHistGradientBoosting(classifier, X_test,save_results=False):
    """
    @brief Predicts the test dataset with the Hist Gradient Boosting Classifier.


    @param classifier The trained Hist Gradient Boosting Classifier.
    @param X_test The test dataset.
    @return The predicted values for the test dataset.
    """
    logging.debug("Predicting test dataset with Hist Gradient Boosting Classifier")

    processed_X_test = preProcessDatasetHistGradientBoosting(X_test)
    if save_results:
        logging.info(f"Saving predicted target variable at {__HIST_GRADIENT_BOOSTING_PREDICTED_CSV_PATH}")

    predictions = predictTestDataset(classifier, processed_X_test, __HIST_GRADIENT_BOOSTING_PREDICTED_CSV_PATH if save_results else None)
    return predictions
