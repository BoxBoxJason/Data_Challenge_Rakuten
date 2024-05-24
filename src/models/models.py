
from pandas import DataFrame


def predictTestDataset(model, processed_X_test, results_path=None):
    """
    @brief Predicts the target variable using the given model and test data.

    This function predicts the target variable using the given model and test data.
    The predictions are saved to a csv file if the results_path is provided.

    @param model The trained model.
    @param processed_X_test The test data.
    @param results_path The path to save the predictions.
    @return The predictions.
    """
    prediction = model.predict(processed_X_test)
    if results_path:
        prediction_df = DataFrame(prediction)
        prediction_df.to_csv(results_path, index=True)
    return prediction


def trainClassifier(model_class, model_params, processed_X_train, y_train):
    """
    @brief Trains a classifier model on the training data.

    This function trains a classifier model on the training data.

    @param model The classifier model.
    @param processed_X_train The training data.
    @param y_train The target variable.
    @return The trained classifier model.
    """
    model = model_class(**model_params, n_jobs=-1, verbose=2)
    model.fit(processed_X_train, y_train)
    return model
