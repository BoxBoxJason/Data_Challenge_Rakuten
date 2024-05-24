import logging
from os.path import join
from os import getenv, makedirs
from sklearn.ensemble import RandomForestClassifier
from models.models import predictTestDataset, trainClassifier, optimizeModelParameters

# Random Forest Classifier results path
__RANDOM_FOREST_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'random_forest_results')
# Random forest predicted csv path
__RANDOM_FOREST_PREDICTED_CSV_PATH = join(__RANDOM_FOREST_RESULTS_PATH, 'predicted_Y_test.csv')

makedirs(__RANDOM_FOREST_RESULTS_PATH, exist_ok=True)


def preProcessDataset(dataset):
    """
    @brief Preprocesses the dataset.


    @param dataset The dataset to be preprocessed.
    @return The preprocessed dataset.
    """
    logging.debug("Preprocessing dataset for Random Forest Classifier")
    return dataset


def optimizeRandomForestClassifierParameters(X_train, y_train):
    """
    @brief Trains a Random Forest Classifier model on the training data.

    This function trains a Random Forest Classifier model on the training data.
    The model is trained using GridSearchCV to find the best hyperparameters.

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Random Forest Classifier model.
    """
    param_grid = {
        'n_estimators': [100, 200, 300,400,500],
        'max_depth': [None,10, 20, 30],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    processed_X_train = preProcessDataset(X_train)

    return optimizeModelParameters(RandomForestClassifier,'Random Forest Classifier', param_grid, __RANDOM_FOREST_RESULTS_PATH,processed_X_train, y_train)


def trainRandomForestClassifier(X_train, y_train, model_params={}):
    """
    @brief Trains a Random Forest Classifier model on the training data.

    This function trains a Random Forest Classifier model on the training data.
    The model is trained using the best hyperparameters found by optimizeRandomForestRegressorParameters().

    @param X_train The training data.
    @param y_train The target variable.
    @return The trained Random Forest Classifier model.
    """
    logging.debug("Training Random Forest Classifier model")
    processed_X_train = preProcessDataset(X_train)
    trained_rf_model = trainClassifier(RandomForestClassifier, model_params, processed_X_train, y_train)
    return trained_rf_model


def predictRandomForestClassifier(rf, X_test, save_results=False):
    """
    @brief Predicts the target variable using a trained Random Forest Classifier model.

    This function predicts the target variable using a trained Random Forest Classifier model.

    @param rf The trained Random Forest Classifier model.
    @param X_test The test data.
    @return The predicted target variable.
    """
    logging.debug("Predicting test dataset for Random Forest Classifier")
    processed_X_test = preProcessDataset(X_test)
    if save_results:
        logging.info(f"Saving predicted results at {__RANDOM_FOREST_PREDICTED_CSV_PATH}")
    prediction = predictTestDataset(rf, processed_X_test, __RANDOM_FOREST_PREDICTED_CSV_PATH if save_results else None)
    return prediction
