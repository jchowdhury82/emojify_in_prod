import os
import pickle

current_base_dir = os.path.dirname(os.path.abspath(__file__))
word_to_index_file = os.path.join(current_base_dir, 'data', 'word_to_index.pickle')
word_to_vec_map_file = os.path.join(current_base_dir, 'data', 'word_to_vec_map.pickle')
model_json_file = os.path.join(current_base_dir, 'models', 'model_metadata.json')
model_weights_file = os.path.join(current_base_dir, 'models', 'model_weights.h5')
max_len=10
with open(word_to_index_file, 'rb') as handle1:
    word_to_index = pickle.load(handle1)
with open(word_to_vec_map_file, 'rb') as handle2:
    word_to_vec_map = pickle.load(handle2)