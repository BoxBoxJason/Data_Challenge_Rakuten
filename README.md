# Data Challenge Rakuten

## WARNING
This repository contains the solutions to the data challenge of Rakuten. If you do not want to see the solutions, please do not read the content of this repository.

## Description
This repository contains the code used to solve the data challenge of Rakuten. The code is written in Python and uses Machine Learning models to predict the target variable. The code is structured in a way that it can be run with different actions to optimize the hyperparameters of the models and predict the target variable.

The goal of the challenge is to predict the type of product based on the product description. It is a multi-class classification problem.

The code uses the following models to predict the target variable:
- Random Forest
- Gradient Boosting
- Histogram Gradient Boosting
- Support Vector Classifier
- Linear Support Vector Classifier
- Nu Support Vector Classifier
- K-Nearest Neighbors
- Gaussian Naive Bayes
- Multinomial Naive Bayes
- Complement Naive Bayes
- Bernoulli Naive Bayes
- AdaBoost
- Bagging
- Extra Trees

The models are evaluated using the weighted F1 score, which represents the harmonic mean of the precision and recall of the model.
We are trying to see which model performs the best on the dataset.


## Getting Started

### Prerequisites
- Python 3.12.0 or higher
- Pip

### Run the code
1. Clone the repository: `git clone https://github.com/BoxBoxJason/Data_Challenge_Rakuten.git`
2. Navigate to the repository: `cd Data_Challenge_Rakuten`
3. Install the required python packages: `pip install -r requirements.txt`
4. Run the code: `python src [ACTIONS]`

### Actions
The code can be run with the following actions (concurently):
- Optimize
    - `optimize_rf`: Optimize the hyperparameters of the Random Forest model
    - `optimize_gb`: Optimize the hyperparameters of the Gradient Boosting model
    - `optimize_hgb`: Optimize the hyperparameters of the Histogram Gradient Boosting model
    - `optimize_svc`: Optimize the hyperparameters of the Support Vector Classifier model
    - `optimize_linear_svc`: Optimize the hyperparameters of the Linear Support Vector Classifier model
    - `optimize_nu_svc`: Optimize the hyperparameters of the Nu Support Vector Classifier model
    - `optimize_knn`: Optimize the hyperparameters of the K-Nearest Neighbors model
    - `optimize_gnb`: Optimize the hyperparameters of the Gaussian Naive Bayes model
    - `optimize_mnb`: Optimize the hyperparameters of the Multinomial Naive Bayes model
    - `optimize_cnb`: Optimize the hyperparameters of the Complement Naive Bayes model
    - `optimize_bnb`: Optimize the hyperparameters of the Bernoulli Naive Bayes model
    - `optimize_ab`: Optimize the hyperparameters of the AdaBoost model
    - `optimize_bagging`: Optimize the hyperparameters of the Bagging model
    - `optimize_et`: Optimize the hyperparameters of the Extra Trees model

- Train & Predict
    - `predict_rf`: Predict the target variable with the Random Forest model
    - `predict_gb`: Predict the target variable with the Gradient Boosting model
    - `predict_hgb`: Predict the target variable with the Histogram Gradient Boosting model
    - `predict_svc`: Predict the target variable with the Support Vector Classifier model
    - `predict_linear_svc`: Predict the target variable with the Linear Support Vector Classifier model
    - `predict_nu_svc`: Predict the target variable with the Nu Support Vector Classifier model
    - `predict_knn`: Predict the target variable with the K-Nearest Neighbors model
    - `predict_gnb`: Predict the target variable with the Gaussian Naive Bayes model
    - `predict_mnb`: Predict the target variable with the Multinomial Naive Bayes model
    - `predict_cnb`: Predict the target variable with the Complement Naive Bayes model
    - `predict_bnb`: Predict the target variable with the Bernoulli Naive Bayes model
    - `predict_ab`: Predict the target variable with the AdaBoost model
    - `predict_bagging`: Predict the target variable with the Bagging model
    - `predict_et`: Predict the target variable with the Extra Trees model

The Optimize action will optimize the hyperparameters of the model and store the results as a json file for a model.
Then the Train & Predict action will train the model with the optimized hyperparameters and predict the target variable. The predictions will be stored as a csv file.

All the data generated is stored in the results directory. These processing takes a very long time to run, hence we already ran them and stored the results in the results directory.
You can also find the predictions output in the artifacts of the pipelines that were run in the GitHub Actions.
