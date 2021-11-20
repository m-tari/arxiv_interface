import os

input_file = 'arxiv_20krows_train.pkl'

input_dir = 'input'
model_dir = 'models'

SRC_PATH = os.path.dirname(os.getcwd())
INPUT_FILE_PATH = os.path.join(SRC_PATH, input_dir, input_file)
MODEL_OUTPUT_PATH = os.path.join(SRC_PATH, model_dir)


features = 'abstract'
labels = 'general_category'