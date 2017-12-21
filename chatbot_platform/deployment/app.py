import flask
import logging
from flask import Flask, request
import sys
import nltk

from chatbot_platform.src.scoring import chat,load_chatbot_model_s3
from chatbot_platform.src.training import train_and_save_chatbot_model_s3

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
app = Flask(__name__)


global imdb_recsys
nltk.download('punkt')
app.chatbot = load_chatbot_model_s3()


@app.route('/')
def heart_beat():
    return flask.jsonify({"status": "ok"})


@app.route('/api/v1/schemas/train', methods=['POST'])
def train():
    app.logger.info("Submitting the training job")
    input_json = request.get_json()
    train_and_save_chatbot_model_s3()
    response = {"message": "Training done!!!"}
    return flask.jsonify(response)


@app.route('/api/v1/schemas/score', methods=['POST'])
def score():
    input_json = request.get_json()
    query = input_json.get("query")
    result = chat(app.chatbot, query)
    return flask.jsonify(result)


if __name__ == "__main__":
    app.run()
