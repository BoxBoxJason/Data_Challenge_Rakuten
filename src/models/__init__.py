from os import getenv, makedirs
from os.path import join

makedirs(getenv('PROJECT_RESULTS_DIR'), exist_ok=True)
