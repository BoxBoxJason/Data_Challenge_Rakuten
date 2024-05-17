from os import getenv,makedirs

makedirs(getenv('PROJECT_LOG_DIR'), exist_ok=True)
makedirs(getenv('PROJECT_RESULTS_DIR'), exist_ok=True)
