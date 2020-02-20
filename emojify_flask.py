# Flask app to call emojify predict function

from flask import Flask, request,render_template
from emojify_model import predict
import config
from keras.models import model_from_json

app = Flask(__name__)

# Load the model to cache for quicker access

with open(config.model_json_file, "r") as handle:
    loaded_model_json = handle.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(config.model_weights_file)

# Create three routes
# 1 - emoapi :  This is for API calls
# 2 - <blank> : This is for a web form to enter the input sentence
# 3 - emojify : This is to respond with prediction.

@app.route("/emoapi", methods = ['GET'])
def emoapi():

    X = request.args.get('input_sentence')
    Y = predict(X,loaded_model)
    return X + ' ' + Y

# To render html form for entering the input sentence
@app.route('/')
def home():
   return render_template('input.html')

# To render output as the predicted emoji
@app.route("/emojify", methods = ['POST'])
def index():

    X = request.form['Thoughts']
    Y = predict(X,loaded_model)
    result = X + ' ' + Y

    return render_template("input.html", emojified=result)


if __name__ == "__main__":
    app.run(threaded=False,debug=False)