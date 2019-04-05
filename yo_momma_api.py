import os
import json
import logging
import boto3
from flask import Flask
from textgenrnn import textgenrnn

# Init flask app
app = Flask(__name__)

# Globals for bucket
bucket = "yo-momma-api"
key_weights = "textgenrnn_weights_yo.hdf5"
key_vocab = "textgenrnn_vocab_yo.json"
key_config = "textgenrnn_config_yo.json"
lambda_loc = '/tmp/'
have_model_files = os.path.isfile(key_weights) and os.path.isfile(
    key_vocab) and os.path.isfile(key_config)


# Download weights
if not have_model_files:
    logging.info('downloading')
    s3_bucket = boto3.resource('s3').Bucket(bucket)
    s3_bucket.download_file(key_weights, lambda_loc + key_weights)
    s3_bucket.download_file(key_vocab, lambda_loc + key_vocab)
    s3_bucket.download_file(key_config, lambda_loc + key_config)


# Load model
textgen = textgenrnn(weights_path=key_weights,
                     vocab_path=key_vocab,
                     config_path=key_config)


@app.route('/ai')
def gen_jokes():
    try:
        texts = textgen.generate(temperature=0.5,
                                 return_as_list=True
                                 )
        text = texts[0]
    except Exception as e:
        logging.error(e)
        text = """Yo' momma so stupid she had a 500 error
                on this api and can't generate more
                jokes now, sorry :0"""
    return json.dumps({'joke': text})


@app.route('/')
def index():
    return json.dumps({'message': 'api active'})


def train():
    textgen_train = textgenrnn()
    textgen_train.train_from_file(
        'jokes.txt',
        new_model=True,
        num_epochs=500,
        train_size=0.9,
        dropout=0.1)


if __name__ == '__main__':
    app.run(threaded=False)
