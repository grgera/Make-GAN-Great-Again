import torch


DEBUG = False 
LIGHT_VER = False 
NO_SAVE = False 

START_TOKEN = '<S>'
END_TOKEN = '<E>'
PAD_TOKEN = '<P>'

CUDA = torch.cuda.is_available()

TRAIN_SIZE = 53000
TEST_SIZE = 3000
VALID_SET_SIZE_RATIO = 0.1

dataset_path = './dataset/quora_duplicate_questions.tsv'