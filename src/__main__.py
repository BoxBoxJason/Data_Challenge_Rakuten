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
from models.extra_trees import optimizeExtraTreesClassifierParameters, trainAndTestExtraTreesClassifier
from models.adaboost import optimizeAdaBoostClassifierParameters, trainAndTestAdaBoostClassifier
from models.bagging import optimizeBaggingClassifierParameters, trainAndTestBaggingClassifier

setupCustomLogger('DEBUG')
X_train, X_test, y_train = prepareDatasets(os.environ['PROJECT_RAW_DATA_DIR'])

if 'optimize_rf' in sys.argv:
    optimizeRandomForestClassifierParameters(X_train, y_train)

if 'optimize_gb' in sys.argv:
    optimizeGradientBoostingClassifierParameters(X_train, y_train)

if 'optimize_hgb' in sys.argv:
    optimizeHistGradientBoostingClassifierParameters(X_train, y_train)

if 'optimize_knn' in sys.argv:
    optimizeKNeighborsClassifierParameters(X_train, y_train)

if 'optimize_gnb' in sys.argv:
    optimizeNaiveBayesGaussianClassifierParameters(X_train, y_train)

if 'optimize_mnb' in sys.argv:
    optimizeMultinomialNaiveBayesClassifierParameters(X_train, y_train)

if 'optimize_cnb' in sys.argv:
    optimizeComplementNaiveBayesClassifierParameters(X_train, y_train)

if 'optimize_bnb' in sys.argv:
    optimizeBernoulliNaiveBayesClassifierParameters(X_train, y_train)

if 'optimize_svc' in sys.argv:
    optimizeSVCParameters(X_train, y_train)

if 'optimize_linear_svc' in sys.argv:
    optimizeLinearSVCParameters(X_train, y_train)

if 'optimize_nu_svc' in sys.argv:
    optimizeNuSVCParameters(X_train, y_train)

if 'optimize_et' in sys.argv:
    optimizeExtraTreesClassifierParameters(X_train, y_train)

if 'optimize_ab' in sys.argv:
    optimizeAdaBoostClassifierParameters(X_train, y_train)

if 'optimize_bagging' in sys.argv:
    optimizeBaggingClassifierParameters(X_train, y_train)

if 'predict_rf' in sys.argv:
    trainAndTestRandomForestClassifier(X_train, y_train, X_test)

if 'predict_gb' in sys.argv:
    trainAndTestGradientBoostingClassifier(X_train, y_train, X_test)

if 'predict_hgb' in sys.argv:
    trainAndTestHistGradientBoostingClassifier(X_train, y_train, X_test)

if 'predict_knn' in sys.argv:
    trainAndTestKNeighborsClassifier(X_train, y_train, X_test)

if 'predict_gnb' in sys.argv:
    trainAndTestNaiveBayesGaussianClassifier(X_train, y_train, X_test)

if 'predict_mnb' in sys.argv:
    trainAndTestMultinomialNaiveBayesClassifier(X_train, y_train, X_test)

if 'predict_cnb' in sys.argv:
    trainAndTestComplementNaiveBayesClassifier(X_train, y_train, X_test)

if 'predict_bnb' in sys.argv:
    trainAndTestBernoulliNaiveBayesClassifier(X_train, y_train, X_test)

if 'predict_svc' in sys.argv:
    trainAndTestSVC(X_train, y_train, X_test)

if 'predict_linear_svc' in sys.argv:
    trainAndTestLinearSVC(X_train, y_train, X_test)

if 'predict_nu_svc' in sys.argv:
    trainAndTestNuSVC(X_train, y_train, X_test)

if 'predict_et' in sys.argv:
    trainAndTestExtraTreesClassifier(X_train, y_train, X_test)

if 'predict_ab' in sys.argv:
    trainAndTestAdaBoostClassifier(X_train, y_train, X_test)

if 'predict_bagging' in sys.argv:
    trainAndTestBaggingClassifier(X_train, y_train, X_test)
