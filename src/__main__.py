import os

os.environ['PROJECT_ROOT_DIR'] = os.path.dirname(os.path.dirname(__file__))
os.environ['PROJECT_DATA_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'assets', 'data')
os.environ['PROJECT_LOG_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'logs')
os.environ['PROJECT_RESULTS_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'results')

from utils.logger import setupCustomLogger
from data.prepare import showDatasetInfo, prepareDatasets
from models.random_forest import optimizeRandomForestRegressorParameters

setupCustomLogger('DEBUG')
X_train, X_test, y_train = prepareDatasets(os.environ['PROJECT_DATA_DIR'])

showDatasetInfo(X_train)
showDatasetInfo(y_train)

optimal_random_forest_regressor = optimizeRandomForestRegressorParameters(X_train, y_train)
