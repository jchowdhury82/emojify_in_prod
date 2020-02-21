import os
from zipfile import ZipFile
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

current_base_dir = os.path.dirname(os.path.abspath(__file__))


def read_glove_vecs(glove_file):
    with ZipFile(glove_file, 'r') as zip:
        with zip.open('glove.6B.50d.txt') as f:
            words = set()
            word_to_vec_map = {}
            for line in f.readlines():
                line = line.decode("utf-8").strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

            i = 1
            words_to_index = {}
            index_to_words = {}
            for w in sorted(words):
                words_to_index[w] = i
                index_to_words[i] = w
                i = i + 1
    return words_to_index, index_to_words, word_to_vec_map



glove_file = os.path.join(current_base_dir, 'data', 'glove6b50dtxt.zip')
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(glove_file)
model_json_file = os.path.join(current_base_dir, 'models', 'model_metadata.json')
model_weights_file = os.path.join(current_base_dir, 'models', 'model_weights.h5')
max_len=10

