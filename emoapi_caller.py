# API Tester
import requests

url = 'http://0.0.0.0:5000/emoapi'
params = {'input_sentence': 'i hate you'}

try:
    response = requests.get(url = url, params = params)
    print(response.text)

except requests.exceptions.ConnectionError as err:
    print('Error connecting to API :' + str(err))