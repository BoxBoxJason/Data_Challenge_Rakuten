import os

os.environ['PROJECT_ROOT_DIR'] = os.path.dirname(os.path.dirname(__file__))
os.environ['PROJECT_DATA_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'assets', 'data')
os.environ['PROJECT_LOG_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'logs')

from utils.logger import setupCustomLogger
from data.prepare import openDatasets

setupCustomLogger('DEBUG')
training_data, test_data, product_types = openDatasets(os.environ['PROJECT_DATA_DIR'])