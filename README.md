# Emojify - from Jupyter Notebook to Docker Container

A sample production implementation of an emojify api based on a NLP model using RNN (LSTM).
Idea and architecture of the model is taken from the labs of the coursera specialization course "Deep Learning" by Prof Andrew Ng.
The model was initially built,trained and tested on a Jupyter Notebook.
This project was meant to put the model to production as an API. 


### Emojification - what it is?

Emojification is attaching an emoji to a sentence that is entered by the user. 

The following emojis are used for our project for basic emojification:
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

### Directory structure

./emojify
   /models
        model_metadata.json     # stores the keras model metadata
        model_weights.h5        # stores weights 
   /data
        data files              # any train/test data files required for testing should be under this folder
        glove embed files       # glove embedding file (zipped)
   /templates
        input.html              # html form for testing from ui
    

    config.py                   # importing all the necessary prerequisites for the model, loading the model

    emojify_flask.py            # flask application to process api routes and call Predict function
    emoapi_caller.py            # python code snippet to call the api
    emojify_model.py            # main python module that has train, test and predict functions
    pickle_glove_embeddings.py  # module for generating word to vector map dictionaries that are required by main module
    
    Dockerfile                  # dockerfile 
    requirements.txt            # python requirements file

    README.md

### Deployment in Docker

A Flask app is created to support access in two ways :
  > As a web form to take user input (single sentence) and show emojified versioning
  > As an API exposing the prediction function for an input sentence

The app is then boxed into a Docker image and pushed to docker hub repository.

Image details:

    Image name - jchowdhury/emojify:v1
    Base image - python:3.6-slim-buster

    Building docker image: docker build -t jchowdhury/emojify:v1 .
    Running docker image : docker run -d -p 5000:5000 --name emojify  jchowdhury/emojify:v1

## Steps to test

1. docker pull jchowdhury/emojify:v1
2. docker run -d -p 5000:5000 --name emojify  jchowdhury/emojify:v1
3. check http://localhost:5000 from browser
4. curl  -X GET  http://127.0.0.1:5000/emoapi -d "input_sentence=hi%20there%20buddy”


## Built With

* Python 3.6 with Keras and Flask
* Docker with base image as Python

## Authors

* Joyjit Chowdhury



## Acknowledgments

* Prof Andrew Ng and Coursera - Deep  Learning specialization

