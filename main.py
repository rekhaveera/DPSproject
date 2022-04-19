from app import app
import pickle
from flask import Flask, request, jsonify, send_from_directory,render_template
import numpy as np
app = Flask(__name__)
model = pickle.load(open('float64model.pkl', 'rb'))

import base64
import hashlib
from Cryptodome.Cipher import AES
from Cryptodome.Random import get_random_bytes

__key__ = hashlib.sha256(b'16-character key').digest()

def encrypt(raw):
    BS = AES.block_size
    pad = lambda s: s + (BS - len(s) % BS) * chr(BS - len(s) % BS)
    raw = base64.b64encode(pad(raw).encode('utf8'))
    iv = b'f5FI\t\xcau\x16gq\x9f_\xb7\xc3\xc3t'
    cipher = AES.new(key= __key__, mode= AES.MODE_CFB,iv= iv)
    return base64.b64encode(iv + cipher.encrypt(raw))

def byte2int(row):
    return int.from_bytes(row, byteorder='little')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',  methods=['POST', 'GET'])
def predict():
    if request.method == "POST":
        int_features = []
        for x in request.form.values():
            int_features.append(x)
        # ['1', '2', '3', '4', '5', '6', 'Yes', 'Male', 'Yes']
        for i in range(6):
            int_features[i] = int(int_features[i])

        int_features[0] = np.nan_to_num(np.array(byte2int(encrypt(str(int_features[0])))).astype(np.float32))
        int_features[2] = np.nan_to_num(np.array(byte2int(encrypt(str(int_features[1])))).astype(np.float32))

        if int_features[6] == "Yes":
            int_features[6] = 1
        else:
            int_features[6] = 0

        if int_features[7] == "Yes":
            int_features[7] = 1
        else:
            int_features[7] = 0

        if int_features[8] == "Male":
            int_features[8] = 1
        else:
            int_features[8] = 0

        if int_features[8] == 1:
            int_features.append(0)
        else:
            int_features.append(1)

        final_features = np.array([(int_features)])
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        return render_template('output.html', prediction_text='Predicted Output {}'.format(output))




if __name__ == "__main__":
    app.run(port=5004, debug=True)