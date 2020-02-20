import numpy as np
import os
import pickle

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
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


current_base_dir = os.path.dirname(os.path.abspath(__file__))
glove_file = os.path.join(current_base_dir, 'data', 'glove.6B.50d.txt')
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs(glove_file)

with open(os.path.join(current_base_dir, 'data', 'word_to_index.pickle'), 'wb') as handle:
    pickle.dump(word_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(current_base_dir, 'data', 'index_to_word.pickle'), 'wb') as handle:
    pickle.dump(index_to_word, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(os.path.join(current_base_dir, 'data', 'word_to_vec_map.pickle'), 'wb') as handle:
    pickle.dump(word_to_vec_map, handle, protocol=pickle.HIGHEST_PROTOCOL)