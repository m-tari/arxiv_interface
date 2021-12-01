import os

input_file = 'sample_df_2021.csv'
model_file = 'classifier_model.bin'

input_dir = 'input'
model_dir = 'models'


ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
SRC_PATH = os.path.abspath(os.path.dirname(__file__))
INPUT_FILE_PATH = os.path.join(ROOT_PATH, input_dir, input_file)
MODEL_OUTPUT_PATH = os.path.join(ROOT_PATH, model_dir, model_file)


features = 'abstract'
labels = 'general_category'
