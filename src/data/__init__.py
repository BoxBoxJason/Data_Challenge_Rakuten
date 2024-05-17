from os import getenv, makedirs

makedirs(getenv('PROJECT_DATA_DIR'), exist_ok=True)
