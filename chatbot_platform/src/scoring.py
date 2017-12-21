from chatbot_platform.src.chatbot_constants import *
from chatbot_platform.src.config import *
from chatbot_platform.src.chatbot_model import ChatbotModel
from util.data_store.local_filesystem import LocalFileSystem
from util.data_store.s3_data_store import S3DataStore

import random


def chat(chatbot_model,sentence):
    results = chatbot_model.predict(sentence)
    if len(results) >0:
        category = results[0][0]
        confidence = results[0][1]
        response_text = random.choice(chatbot_model.response[category])
    else:
        category = "Unknown"
        confidence = "Unknown"
        response_text = "I dont know what you are talking about."


    response_dict = dict()
    response_dict["category"] = category
    response_dict["confidence"] = str(confidence)
    response_dict["response_text"] = response_text
    return response_dict




def load_chatbot_model_local(src_dir):
    data_store = LocalFileSystem(src_dir=src_dir)
    chatbot = ChatbotModel.load(data_store=data_store)
    return chatbot


def load_chatbot_model_s3():
    data_store = S3DataStore(src_bucket_name=AWS_BUCKET_NAME, access_key=AWS_S3_ACCESS_KEY_ID,
                             secret_key=AWS_S3_SECRET_ACCESS_KEY)
    chatbot = ChatbotModel.load(data_store=data_store)
    return chatbot


def load_credential_local(src_dir):
    data_store = LocalFileSystem(src_dir=src_dir)
    credential_json = data_store.read_json_file(CREDENTIAL_FILENAME)
    return dict(credential_json)

def load_credential_s3():
    data_store = S3DataStore(src_bucket_name=AWS_BUCKET_NAME, access_key=AWS_S3_ACCESS_KEY_ID,
                             secret_key=AWS_S3_SECRET_ACCESS_KEY)
    credential_json = data_store.read_json_file(CREDENTIAL_FILENAME)
    return dict(credential_json)

def verify_credential(credential_dict,username,password):
    if username in credential_dict:
        if password in credential_dict[username]:
            return credential_dict[username][0]
    return None

def get_username_from_userid(credential_dict,user_id):
    result = None
    for username in credential_dict:
        if credential_dict[username][0] == user_id:
            result = username
            break
    return result

def write_incident_json_to_s3(contents, filename):
    data_store = S3DataStore(src_bucket_name=AWS_BUCKET_NAME, access_key=AWS_S3_ACCESS_KEY_ID,
                             secret_key=AWS_S3_SECRET_ACCESS_KEY)
    data_store.write_json_file(contents=contents,filename=INCIDENT_FOLDERNAME+filename)

if __name__ == '__main__':
    # chatbot = load_chatbot_model_local(src_dir="./chatbot_platform/data")
    # query = "my macbook is restarting everytime I try to open MS office"
    # result = chat(chatbot, query)
    # print(result)

    chatbot = load_chatbot_model_s3()
    query = "are you interested in india pakistan war?"
    result = chat(chatbot, query)
    print(result)

