# Emojify

This project uses NLP with LSTM to propose an emoji for a given input sentence.
Idea and architecture is taken from the labs of the coursera specialization course "Deep Learning" by Prof Andrew Ng.



## Getting Started

The following are the details of the files :

### Emojification

The following emojis are used:
    Heart
    Baseball
    Smile
    Disappointed
    Food

Input sentence is assigned one of the above emoji's

Eg:
Input = i love being lazy
Output = i love being lazy ❤️!

Input = This is so miserable
Output = This is so miserable 😞!


### Prerequisites

This needs Glove embedding file glove6b50dtxt which is available in kaggle. https://www.kaggle.com/watts2/glove6b50dtxt

### Design of Model

1. An input sentence is converted to an embedding vector using Glove embedding.
Max length of input sentence is 10.

2. A custome embedding layer is implemented in Keras.
The mapping of a word to its Glove embedding is maintained in a dictionary word_to_vec_map and is stored
as a pickle file. The Glove file reader and embedding pickle file generator is in pickle_glove_embedding.py

3. LSTM is implemented in Keras to build model.
 Input = Sentence with Words
 Layer 1 = Generate word embeddings using customer word embedding above
 Layer 2 = LSTM with 128 units, all hidden units returned to next layer
 Layer 3 = Dropout regularization with p=0.5
 Layer 4 = LSTM with 128 units, only last unit (last time step) returned
 Layer 5 = Dropout egularization with p=0.5
 Layer 6 = Softmax for multiple classification


4. Model is trained with Adam optimizer and Loss = categorical_crossentropy to support multi class classification

5. With a small data set, approx 85% accuracy is achieved.

Model building and training is in the file emojify_lstm_train.py

### Deployment and Use

A Flask app is created to support access in two ways :
1. As a web form to take user input (single sentence) and show emojified versioning
2. As an API exposing the prediction function for an input sentence


The Docker image jc_emojify:1.0 is made available that contains the components.


## Steps to test

1. Download docker image jc_emojify:1.0
2. Use the emojify_caller.py function

## Built With

* Python 3.6 with Keras and Flask
* Docker with base image as Python

## Authors

* Joyjit Chowdhury



## Acknowledgments

* Prof Andrew Ng and Coursera - Deep  Learning specialization

