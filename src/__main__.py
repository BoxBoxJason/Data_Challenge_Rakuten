import os
import sys

os.environ['PROJECT_ROOT_DIR'] = os.path.dirname(os.path.dirname(__file__))
os.environ['PROJECT_RAW_DATA_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'assets', 'data','raw')
os.environ['PROJECT_PROCESSED_DATA_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'assets', 'data','processed')
os.environ['PROJECT_LOG_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'logs')
os.environ['PROJECT_RESULTS_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'results')

from utils.logger import setupCustomLogger
from data.prepare import prepareDatasets
from models.random_forest import optimizeRandomForestClassifierParameters
from models.gradient_boosting import optimizeGradientBoostingClassifierParameters, optimizeHistGradientBoostingClassifierParameters
from models.k_nearest_neighbors import optimizeKNeighborsClassifierParameters
from models.bayesian import optimizeComplementNaiveBayesClassifierParameters, optimizeMultinomialNaiveBayesClassifierParameters, optimizeNaiveBayesGaussianClassifierParameters, optimizeBernoulliNaiveBayesClassifierParameters
from models.support_vector_machines import optimizeSVCParameters, optimizeLinearSVCParameters, optimizeNuSVCParameters

setupCustomLogger('DEBUG')
X_train, X_test, y_train = prepareDatasets(os.environ['PROJECT_RAW_DATA_DIR'])

# Check if optimize random forest in __main__.py arguments
if 'optimize_random_forest' in sys.argv:
    best_params = optimizeRandomForestClassifierParameters(X_train, y_train)

if 'optimize_gradient_boosting' in sys.argv:
    best_params = optimizeGradientBoostingClassifierParameters(X_train, y_train)

if 'optimize_hist_gradient_boosting' in sys.argv:
    best_params = optimizeHistGradientBoostingClassifierParameters(X_train, y_train)

if 'optimize_k_nearest_neighbors' in sys.argv:
    best_params = optimizeKNeighborsClassifierParameters(X_train, y_train)

if 'optimize_naive_bayes_gaussian' in sys.argv:
    best_params = optimizeNaiveBayesGaussianClassifierParameters(X_train, y_train)

if 'optimize_naive_bayes_multinomial' in sys.argv:
    best_params = optimizeMultinomialNaiveBayesClassifierParameters(X_train, y_train)

if 'optimize_naive_bayes_complement' in sys.argv:
    best_params = optimizeComplementNaiveBayesClassifierParameters(X_train, y_train)

if 'optimize_naive_bayes_bernouilli' in sys.argv:
    best_params = optimizeBernoulliNaiveBayesClassifierParameters(X_train, y_train)

if 'optimize_svc' in sys.argv:
    best_params = optimizeSVCParameters(X_train, y_train)

if 'optimize_linear_svc' in sys.argv:
    best_params = optimizeLinearSVCParameters(X_train, y_train)

if 'optimize_nu_svc' in sys.argv:
    best_params = optimizeNuSVCParameters(X_train, y_train)
