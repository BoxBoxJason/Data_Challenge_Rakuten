import os
import sys

os.environ['PROJECT_ROOT_DIR'] = os.path.dirname(os.path.dirname(__file__))
os.environ['PROJECT_RAW_DATA_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'assets', 'data','raw')
os.environ['PROJECT_PROCESSED_DATA_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'assets', 'data','processed')
os.environ['PROJECT_LOG_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'logs')
os.environ['PROJECT_RESULTS_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'results')

from utils.logger import setupCustomLogger
from data.prepare import prepareDatasets
from models.random_forest import optimizeRandomForestClassifierParameters, trainAndTestRandomForestClassifier
from models.gradient_boosting import optimizeGradientBoostingClassifierParameters, optimizeHistGradientBoostingClassifierParameters, \
    trainAndTestGradientBoostingClassifier, trainAndTestHistGradientBoostingClassifier
from models.k_nearest_neighbors import optimizeKNeighborsClassifierParameters, trainAndTestKNeighborsClassifier
from models.bayesian import optimizeComplementNaiveBayesClassifierParameters, optimizeMultinomialNaiveBayesClassifierParameters, \
    optimizeNaiveBayesGaussianClassifierParameters, optimizeBernoulliNaiveBayesClassifierParameters, \
    trainAndTestNaiveBayesGaussianClassifier, trainAndTestMultinomialNaiveBayesClassifier, trainAndTestComplementNaiveBayesClassifier, \
    trainAndTestBernoulliNaiveBayesClassifier
from models.support_vector_machines import optimizeSVCParameters, optimizeLinearSVCParameters, optimizeNuSVCParameters, \
    trainAndTestSVC, trainAndTestLinearSVC, trainAndTestNuSVC

setupCustomLogger('DEBUG')
X_train, X_test, y_train = prepareDatasets(os.environ['PROJECT_RAW_DATA_DIR'])

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

if 'optimize_naive_bayes_bernoulli' in sys.argv:
    best_params = optimizeBernoulliNaiveBayesClassifierParameters(X_train, y_train)

if 'optimize_svc' in sys.argv:
    best_params = optimizeSVCParameters(X_train, y_train)

if 'optimize_linear_svc' in sys.argv:
    best_params = optimizeLinearSVCParameters(X_train, y_train)

if 'optimize_nu_svc' in sys.argv:
    best_params = optimizeNuSVCParameters(X_train, y_train)

if 'predict_random_forest' in sys.argv:
    trainAndTestRandomForestClassifier(X_train, y_train, X_test)

if 'predict_gradient_boosting' in sys.argv:
    trainAndTestGradientBoostingClassifier(X_train, y_train, X_test)

if 'predict_hist_gradient_boosting' in sys.argv:
    trainAndTestHistGradientBoostingClassifier(X_train, y_train, X_test)

if 'predict_k_nearest_neighbors' in sys.argv:
    trainAndTestKNeighborsClassifier(X_train, y_train, X_test)

if 'predict_naive_bayes_gaussian' in sys.argv:
    trainAndTestNaiveBayesGaussianClassifier(X_train, y_train, X_test)

if 'predict_naive_bayes_multinomial' in sys.argv:
    trainAndTestMultinomialNaiveBayesClassifier(X_train, y_train, X_test)

if 'predict_naive_bayes_complement' in sys.argv:
    trainAndTestComplementNaiveBayesClassifier(X_train, y_train, X_test)

if 'predict_naive_bayes_bernoulli' in sys.argv:
    trainAndTestBernoulliNaiveBayesClassifier(X_train, y_train, X_test)

if 'predict_svc' in sys.argv:
    trainAndTestSVC(X_train, y_train, X_test)

if 'predict_linear_svc' in sys.argv:
    trainAndTestLinearSVC(X_train, y_train, X_test)

if 'predict_nu_svc' in sys.argv:
    trainAndTestNuSVC(X_train, y_train, X_test)
