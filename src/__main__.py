import os

os.environ['PROJECT_ROOT_DIR'] = os.path.dirname(os.path.dirname(__file__))
os.environ['PROJECT_DATA_DIR'] = os.path.join(os.environ['PROJECT_ROOT_DIR'], 'assets', 'data')

from data.prepare import openDatasets

training_data, test_data, product_types = openDatasets(os.environ['PROJECT_DATA_DIR'])