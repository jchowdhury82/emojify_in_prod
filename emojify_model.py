# File with functions to train / test model
# Using Keras for LSTM based emotion determination


import numpy as np
import sys
import emoji
import csv
import config
import os


from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.models import model_from_json




#We are using just 5 emojis
emoji_dictionary = {"0": "\u2764\uFE0F",  # :heart: prints a black instead of red heart depending on the font
                    "1": ":baseball:",
                    "2": ":smile:",
                    "3": ":disappointed:",
                    "4": ":fork_and_knife:"}

def read_csv(filename):
    phrase = []
    emoji = []

    with open(filename) as csvDataFile:
        csvReader = csv.reader(csvDataFile)

        for row in csvReader:
            phrase.append(row[0])
            emoji.append(row[1])

    X = np.asarray(phrase)
    Y = np.asarray(emoji, dtype=int)

    return X, Y


def label_to_emoji(label):
    return emoji.emojize(emoji_dictionary[str(label)], use_aliases=True)


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)]
    return Y


def sentences_to_indices(X):
    m = X.shape[0]  # number of training examples
    X_indices = np.zeros((m, config.max_len))

    for i in range(m):  # loop over training examples
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            X_indices[i, j] = config.word_to_index[w]
            j = j + 1

    return X_indices

# Creation of embedding layer in Keras to convert the input data
# Embedding layer is based on Glove file obtained from Kaggle
def create_embedding_layer():

    vocab_len = len(config.word_to_index) + 1
    emb_dim = config.word_to_vec_map["sample"].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))

    for word, idx in config.word_to_index.items():
        emb_matrix[idx, :] = config.word_to_vec_map[word]


    embedding_layer = Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable=False, input_length=None)
    embedding_layer.build((None,))
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer


# Build model layers
# Input = Sentence with Words
# Layer 1 = Generate word embeddings using customer word embedding above
# Layer 2 = LSTM with 128 units, all hidden units returned to next layer
# Layer 3 = Dropout regularization with p=0.5
# Layer 4 = LSTM with 128 units, only last unit (last time step) returned
# Layer 5 = Dropout egularization with p=0.5
# Layer 6 = Softmax for multiple classification

def build_model(input_shape):

    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = create_embedding_layer()
    embeddings = embedding_layer(sentence_indices)
    X = LSTM(units=128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(units=128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(units=5)(X)
    X = Activation('softmax')(X)

    model = Model(inputs=sentence_indices, outputs=X)

    return model

# Function to train / test model
# Uses LOSS = categorical cross entropy  and optimizer = ADAM
# Model metadata and weights are stored separately
# Train is done in X_train / Y_train
# Test is done in X_test / Y_test
# Input Labels are converted to Onehot vectors with 5 classes corresponding to the 5 emoji's

def train_model(X_train,Y_train):

    try:
        # train model on train data

        X_train_indices = sentences_to_indices(X_train)
        Y_train_oh = convert_to_one_hot(Y_train, C=5)

        model = build_model((config.max_len,))

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train_indices, Y_train_oh, epochs=50, batch_size=32, shuffle=True, verbose=0)

        # serialize model to JSON,weights to HDF5
        model_json = model.to_json()
        with open(config.model_json_file, "w") as handle:
            handle.write(model_json)
        model.save_weights(config.model_weights_file)

        print("Training completed and model saved in :" + config.model_json_file + " and " + config.model_weights_file)

        return model

    except AssertionError as err:
        print("Error finding files : " + str(err))
        return None

    except:
        print("Error in training/testing model"  + str(sys.exc_info()[0]))
        return None


def test_model(X_test,Y_test):

    try:
        # load model from json and weights from h5 file
        with open(config.model_json_file, "r") as handle:
            loaded_model_json = handle.read()

        model = model_from_json(loaded_model_json)
        model.load_weights(config.model_weights_file)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


        # evaluate model on test data
        X_test_indices = sentences_to_indices(X_test)
        Y_test_oh = convert_to_one_hot(Y_test, C=5)

        score = model.evaluate(X_test_indices, Y_test_oh,verbose = 0)

        evaluation_metric_name = model.metrics_names[1]
        evaluation_metric_score = score[1] * 100

        print("Evaluation completed")

        return evaluation_metric_name,evaluation_metric_score

    except:
        print("Error in training/testing model"  + str(sys.exc_info()[0]))
        return None,0


# Predictor function
def predict(X,loaded_model):

    try:
        assert isinstance(X,str),"Input is not string"
        #convert input from string to numpy array

        X_arr = np.asarray([X])
        X_indices = sentences_to_indices(X_arr)

        emojifiedX = label_to_emoji(np.argmax(loaded_model.predict(X_indices)))

        return emojifiedX

    except AssertionError as err:

        print("Error in prediction : "  + str(err))
        return "<** ERROR **>"

    except:

        print("Error in prediction:" + sys.exc_info()[0])
        return "<** ERROR **>"



'''


# TRAIN  SECTION
train_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','train_emoji.csv')
test_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','test_emoji.csv')

X_train, Y_train = read_csv(train_file)
X_test, Y_test = read_csv(test_file)

try:
    model = train_model(X_train,Y_train)
except:
    print("Error in training model")
    model = None

try:
    metric_name, score = test_model(X_test,Y_test)
    print("%s: %.2f%%" % (metric_name, score))
except:
    print("Error in training model")


X = "i dont like this guy"
Y = predict(X,model)
print(X + ' ' + Y)

'''