import logging
from os.path import join
from os import getenv, makedirs
from sklearn.neighbors import KNeighborsClassifier
from models.models import predictTestDataset, trainClassifier, optimizeModelParameters

# K Neighbors Classifier results path
__K_NEIGHBORS_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'k_nearest_neighbors_results')
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
    param_grid = {
        'n_neighbors': [5, 10, 15],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree'],
        'leaf_size': [20, 30],
        'p': [1, 2]
    }

    processed_X_train = preProcessDataset(X_train)

    return optimizeModelParameters(KNeighborsClassifier,'K-Nearest Neighbors Classifier', param_grid,__K_NEIGHBORS_RESULTS_PATH,processed_X_train,y_train)


def trainKNeighborsClassifier(X_train, y_train, model_params={}):
    """
    @brief Trains a K Neighbors Classifier model on the training data.

    This function trains a K Neighbors Classifier model on the training data.

    @param X_train The training data.
    @param y_train The target variable.
    @param model_params The parameters for the K Neighbors Classifier model.
    @return The trained K Neighbors Classifier model.
    """
    logging.debug("Training K Neighbors Classifier model")

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
