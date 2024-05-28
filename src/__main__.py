import os
import sys

os.environ['PROJECT_ROOT_DIR'] = os.path.dirname(os.path.dirname(__file__))
os.environ['PROJECT_RAW_DATA_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'assets', 'data','raw')
os.environ['PROJECT_PROCESSED_DATA_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'assets', 'data','processed')
os.environ['PROJECT_LOG_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'logs')
os.environ['PROJECT_RESULTS_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'results')

from utils.logger import setupCustomLogger
from data.prepare import prepareDatasets
from models.models import drawValidationScores, drawRealScores
from models.ensemble.random_forest import optimizeRandomForestClassifierParameters, trainAndTestRandomForestClassifier
from models.ensemble.gradient_boosting import optimizeGradientBoostingClassifierParameters, trainAndTestGradientBoostingClassifier
from models.ensemble.extra_trees import optimizeExtraTreesClassifierParameters, trainAndTestExtraTreesClassifier
from models.ensemble.adaboosting import optimizeAdaBoostClassifierParameters, trainAndTestAdaBoostClassifier
from models.ensemble.bagging import optimizeBaggingClassifierParameters, trainAndTestBaggingClassifier
from models.neighbors.k_nearest_neighbors import optimizeKNeighborsClassifierParameters, trainAndTestKNeighborsClassifier
from models.neighbors.radius_neighbors import optimizeRadiusNeighborsClassifierParameters, trainAndTestRadiusNeighborsClassifier
from models.naive_bayes.multinomial import optimizeMultinomialNaiveBayesClassifierParameters, trainAndTestMultinomialNaiveBayesClassifier
from models.naive_bayes.complement import optimizeComplementNaiveBayesClassifierParameters, trainAndTestComplementNaiveBayesClassifier
from models.naive_bayes.bernoulli import optimizeBernoulliNaiveBayesClassifierParameters, trainAndTestBernoulliNaiveBayesClassifier
from models.svm.svc import optimizeSVCParameters, trainAndTestSVC
from models.svm.nu_svc import optimizeNuSVCParameters, trainAndTestNuSVC
from models.svm.linear_svc import optimizeLinearSVCParameters, trainAndTestLinearSVC
from models.linear.ridge import optimizeRidgeParameters, trainAndTestRidgeModel
from models.linear.logistic_regression import optimizeLogisticRegressionParameters, trainAndTestLogisticRegression
from models.linear.sgd import optimizeSGDParameters, trainAndTestSGDModel
from models.linear.passive_aggressive import optimizePassiveAggressiveParameters, trainAndTestPassiveAggressiveModel
from models.linear.perceptron import optimizePerceptronParameters, trainAndTestPerceptron
from models.linear.lasso import optimizeLassoParameters, trainAndTestLasso
from models.linear.elastic_net import optimizeElasticNetParameters, trainAndTestElasticNet
from models.tree.decision_tree import optimizeDecisionTreeParameters, trainAndTestDecisionTreeModel
from models.neural_network.mlp import optimizeMLPClassifierParameters, trainAndTestMLPClassifier


setupCustomLogger('DEBUG')
X_train, X_test, y_train = prepareDatasets(os.environ['PROJECT_RAW_DATA_DIR'])

if 'optimize_rf' in sys.argv:
    optimizeRandomForestClassifierParameters(X_train, y_train)

if 'optimize_gb' in sys.argv:
    optimizeGradientBoostingClassifierParameters(X_train, y_train)

if 'optimize_knn' in sys.argv:
    optimizeKNeighborsClassifierParameters(X_train, y_train)

if 'optimize_rn' in sys.argv:
    optimizeRadiusNeighborsClassifierParameters(X_train, y_train)

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

if 'optimize_ridge' in sys.argv:
    optimizeRidgeParameters(X_train, y_train)

if 'optimize_lr' in sys.argv:
    optimizeLogisticRegressionParameters(X_train, y_train)

if 'optimize_sgd' in sys.argv:
    optimizeSGDParameters(X_train, y_train)

if 'optimize_pa' in sys.argv:
    optimizePassiveAggressiveParameters(X_train, y_train)

if 'optimize_perceptron' in sys.argv:
    optimizePerceptronParameters(X_train, y_train)

if 'optimize_lasso' in sys.argv:
    optimizeLassoParameters(X_train, y_train)

if 'optimize_en' in sys.argv:
    optimizeElasticNetParameters(X_train, y_train)

if 'optimize_dt' in sys.argv:
    optimizeDecisionTreeParameters(X_train, y_train)

if 'optimize_mlp' in sys.argv:
    optimizeMLPClassifierParameters(X_train, y_train)

if 'predict_rf' in sys.argv:
    trainAndTestRandomForestClassifier(X_train, y_train, X_test)

if 'predict_gb' in sys.argv:
    trainAndTestGradientBoostingClassifier(X_train, y_train, X_test)

if 'predict_knn' in sys.argv:
    trainAndTestKNeighborsClassifier(X_train, y_train, X_test)

if 'predict_rn' in sys.argv:
    trainAndTestRadiusNeighborsClassifier(X_train, y_train, X_test)

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

if 'predict_ridge' in sys.argv:
    trainAndTestRidgeModel(X_train, y_train, X_test)

if 'predict_lr' in sys.argv:
    trainAndTestLogisticRegression(X_train, y_train, X_test)

if 'predict_sgd' in sys.argv:
    trainAndTestSGDModel(X_train, y_train, X_test)

if 'predict_pa' in sys.argv:
    trainAndTestPassiveAggressiveModel(X_train, y_train, X_test)

if 'predict_perceptron' in sys.argv:
    trainAndTestPerceptron(X_train, y_train, X_test)

if 'predict_lasso' in sys.argv:
    trainAndTestLasso(X_train, y_train, X_test)

if 'predict_en' in sys.argv:
    trainAndTestElasticNet(X_train, y_train, X_test)

if 'predict_dt' in sys.argv:
    trainAndTestDecisionTreeModel(X_train, y_train, X_test)

if 'predict_mlp' in sys.argv:
    trainAndTestMLPClassifier(X_train, y_train, X_test)
