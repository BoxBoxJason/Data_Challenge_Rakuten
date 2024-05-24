import logging
from os.path import join
from os import getenv, makedirs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from utils.json import saveJson
from models.models import predictTestDataset, trainClassifier

# K Neighbors Classifier results path
__K_NEIGHBORS_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'k_nearest_neighbors_results')
# Best parameters for K Neighbors Classifier path
__K_NEIGHBORS_BEST_PARAMS_PATH = join(__K_NEIGHBORS_RESULTS_PATH, 'best_params.json')
# All tests results for K Neighbors Classifier path
__K_NEIGHBORS_ALL_TESTS_RESULTS_PATH = join(__K_NEIGHBORS_RESULTS_PATH, 'all_tests_results.json')
# K Neighbors predicted csv path
__K_NEIGHBORS_PREDICTED_CSV_PATH = join(__K_NEIGHBORS_RESULTS_PATH, 'predicted_Y_test.csv')

makedirs(__K_NEIGHBORS_RESULTS_PATH, exist_ok=True)

def preProcessDataset(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for K Neighbors Classifier")
    return dataset


def optimizeKNeighborsClassifierParameters(X_train, y_train):
    """
    @brief Trains a K Neighbors Classifier model on the training data.

    This function trains a K Neighbors Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained K Neighbors Classifier model.
    """
    logging.debug("Training K Neighbors Classifier model")
    knb = KNeighborsClassifier()
    param_grid = {
        'n_neighbors': [5, 10, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'leaf_size': [20, 30],
        'p': [1, 2]
    }

    grid_search = GridSearchCV(estimator=knb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='f1_weighted')

    processed_X_train = preProcessDataset(X_train)
    grid_search.fit(processed_X_train, y_train)

    logging.info(f"Best parameters for K Neighbors Classifier: {grid_search.best_params_}")
    logging.info(f"Best score for K Neighbors Classifier: {grid_search.best_score_}")

    logging.info(f"Saving best parameters at {__K_NEIGHBORS_BEST_PARAMS_PATH}")
    saveJson(grid_search.best_params_, __K_NEIGHBORS_BEST_PARAMS_PATH)

    logging.info(f"Saving all tests results at {__K_NEIGHBORS_ALL_TESTS_RESULTS_PATH}")
    saveJson(grid_search.cv_results_, __K_NEIGHBORS_ALL_TESTS_RESULTS_PATH)

    return grid_search.best_estimator_


def trainKNeighborsClassifier(X_train, y_train, model_params=None):
    """
    @brief Trains a K Neighbors Classifier model on the training data.

    This function trains a K Neighbors Classifier model on the training data.

    @param X_train The training data.
    @param y_train The target variable.
    @param model_params The parameters for the K Neighbors Classifier model.
    @return The trained K Neighbors Classifier model.
    """
    logging.debug("Training K Neighbors Classifier model")
    if model_params is None:
        model_params = {}

    processed_X_train = preProcessDataset(X_train)
    model = trainClassifier(KNeighborsClassifier, model_params, processed_X_train, y_train)

    return model


def predictKNeighborsClassifier(k_neighbors, X_test,save_results=False):
    """
    @brief Predicts the target variable using the given model and test data.

    This function predicts the target variable using the given model and test data.
    The predictions are saved to a csv file.

    @param model The trained model.
    @param processed_X_test The test data.
    """
    logging.debug("Predicting test dataset for Random Forest Classifier")
    processed_X_test = preProcessDataset(X_test)
    if save_results:
        logging.info(f"Saving predicted results at {__K_NEIGHBORS_PREDICTED_CSV_PATH}")
    prediction = predictTestDataset(k_neighbors, processed_X_test, __K_NEIGHBORS_PREDICTED_CSV_PATH if save_results else None)
    return prediction
