import logging
from os import makedirs, getenv
from os.path import join
from sklearn.ensemble import StackingClassifier
import joblib
from pandas import DataFrame

# Stacking Classifier results path
__STACKING_RESULTS_PATH = join(getenv('PROJECT_RESULTS_DIR'), 'stacking')

makedirs(__STACKING_RESULTS_PATH, exist_ok=True)

__STACKS_PATH = {
#    "rf": join(getenv('PROJECT_RESULTS_DIR'), 'random_forest','trained_model.pkl'),
    "linear_svc": join(getenv('PROJECT_RESULTS_DIR'), 'linear_svc','trained_model.pkl'),
    "lr": join(getenv('PROJECT_RESULTS_DIR'), 'logistic_regression','trained_model.pkl'),
    "bnb": join(getenv('PROJECT_RESULTS_DIR'), 'bernoulliNB','trained_model.pkl'),
}


def trainAndTestStackingClassifier(X_train, y_train, X_test):
    """
    @brief Trains a Stacking Classifier model on the training data.

    This function trains a Stacking Classifier model on the training data.

    @param X_train The training dataset.
    @param y_train The labels for the training dataset.
    @param X_test The test dataset.
    @return The trained Stacking Classifier and the predictions.
    """
    estimators = []
    logging.debug("Loading estimators.")
    for estimator_name, estimator_path in __STACKS_PATH.items():
        estimators.append((estimator_name, joblib.load(estimator_path)))
    sc = StackingClassifier(estimators=estimators, cv="prefit", n_jobs=-1, passthrough=True,verbose=3)
    logging.info("Training Stacking Classifier with Random Forest, Linear SVC, Logistic Regression, and BernoulliNB.")

    sc.fit(X_train, y_train)

    logging.info(f"Saving trained Stacking Classifier model at {__STACKING_RESULTS_PATH}")
    joblib.dump(sc, join(__STACKING_RESULTS_PATH, 'trained_model.pkl'))

    prediction = sc.predict(X_test)
    prediction_path = join(__STACKING_RESULTS_PATH, 'predicted_Y_test.csv')
    logging.info(f"Saving predictions at {prediction_path}")

    # Set the column name to the target variable name and add offset to the index
    prediction_df = DataFrame(prediction)
    prediction_df.columns = ['prdtypecode']

    prediction_df.index += 84916

    prediction_df.to_csv(prediction_path, index=True)
