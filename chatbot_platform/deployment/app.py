import flask
import logging
from flask import Flask, request
import sys
import nltk
import datetime

from chatbot_platform.src.scoring import chat, load_chatbot_model_s3, load_credential_s3, verify_credential, \
    get_username_from_userid, write_incident_json_to_s3
from chatbot_platform.src.training import train_and_save_chatbot_model_s3
from chatbot_platform.src.chatbot_util import get_unique_number

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
app = Flask(__name__)

global imdb_recsys
nltk.download('punkt')
app.chatbot = load_chatbot_model_s3()
app.credential = load_credential_s3()


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


@app.route('/api/v1/schemas/authenticate', methods=['POST'])
def authenticate():
    input_json = request.get_json()
    username = input_json["username"]
    password = input_json["password"]
    user_id = verify_credential(app.credential, username, password)
    result = dict()
    result["user_id"] = user_id
    return flask.jsonify(result)


@app.route('/api/v1/schemas/raise_incident', methods=['POST'])
def raise_incident():
    input_json = request.get_json()
    user_id = input_json["user_id"]
    priority = input_json["priority"]
    subject = input_json["subject"]
    symptom = input_json["symptom"]
    username = get_username_from_userid(app.credential, user_id)
    result = dict()
    result["user_id"] = user_id
    datetime_value = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    result["datetime"] = datetime_value
    incident_id = get_unique_number()
    result["incident_id"] = incident_id
    result["username"] = username
    result["priority"] = priority
    result["subject"] = subject
    result["symptom"] = symptom
    datetime_value = datetime_value.replace(" ","_")
    write_incident_json_to_s3(contents = result, filename=user_id+datetime_value+".json")
    return_value = {"status":"Incident Created Successfully!!","incident_id":incident_id}
    return flask.jsonify(return_value)


if __name__ == "__main__":
    app.run()
