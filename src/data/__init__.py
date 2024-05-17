from os import getenv, makedirs

makedirs(getenv('PROJECT_RAW_DATA_DIR'), exist_ok=True)
makedirs(getenv('PROJECT_PROCESSED_DATA_DIR'), exist_ok=True)
