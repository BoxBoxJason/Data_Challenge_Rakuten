from os import getenv, makedirs

makedirs(getenv('PROJECT_RESULTS_DIR'), exist_ok=True)
