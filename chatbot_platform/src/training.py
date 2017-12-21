from time import time

from chatbot_platform.src.chatbot_model import ChatbotModel
from chatbot_platform.src.config import *
from util.data_store.local_filesystem import LocalFileSystem
from util.data_store.s3_data_store import S3DataStore


def train_and_save_chatbot_model(data_store):
    chatbot = ChatbotModel.train(data_store)
    chatbot.save(data_store=data_store)


def train_and_save_chatbot_model_local():
    data_store = LocalFileSystem(src_dir="./chatbot_platform/data/")
    train_and_save_chatbot_model(data_store=data_store)


def train_and_save_chatbot_model_s3():
    data_store = S3DataStore(src_bucket_name=AWS_BUCKET_NAME, access_key=AWS_S3_ACCESS_KEY_ID,
                             secret_key=AWS_S3_SECRET_ACCESS_KEY)
    train_and_save_chatbot_model(data_store=data_store)


if __name__ == '__main__':
    t0 = time()
    train_and_save_chatbot_model_local()
    train_and_save_chatbot_model_s3()
    print('running time : ', time() - t0)
