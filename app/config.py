from utils import load_config

# Load configuration
_config = load_config()

# Path variables
MODELS_DIR = _config['paths']['models_dir']
DATA_DIR = _config['paths']['data_dir']
NOTEBOOKS_DIR = _config['paths']['notebooks_dir']
APP_DIR = _config['paths']['app_dir']

# Model paths
TFIDF_VECTORIZER = _config['paths']['tfidf_vectorizer']
SVM_MODEL = _config['paths']['svm_model']
SVM_MODEL_BASIC = _config['paths']['svm_model_basic']

# Data file paths
FAKE_REVIEWS_DATASET = _config['paths']['fake_reviews_dataset']
TEAMS_DATASET = _config['paths']['teams_dataset']

# Notebook path
NOTEBOOK = _config['paths']['notebook']
