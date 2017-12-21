import numpy as np
import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
import tensorflow as tf
import random
from chatbot_platform.src.config import *
from chatbot_platform.src.chatbot_constants import *
from chatbot_platform.src.chatbot_util import *


from util.data_store.local_filesystem import LocalFileSystem
from util.data_store.s3_data_store import S3DataStore



class ChatbotModel(object):
    def __init__(self, dl_model, words, classes, num_input, num_output,response):
        self.dl_model = dl_model
        self.words = words
        self.classes = classes
        self.num_input = num_input
        self.num_output = num_output
        self.response = response

    def save(self, data_store):
        word_class_dict = {"words": self.words, "classes": self.classes, "num_input": self.num_input,
                           "num_output": self.num_output,"response":self.response}
        if type(data_store) is LocalFileSystem:
            data_store.write_dl_model(data=self.dl_model, filename=MODEL_FILENAME)
            data_store.write_pickle_file(data=word_class_dict, filename=WORD_CLASS_DICT_FILENAME)
        if type(data_store) is S3DataStore:
            temp_data_store = LocalFileSystem("/tmp/")
            temp_data_store.write_dl_model(data=self.dl_model, filename=MODEL_FILENAME)
            temp_data_store.write_pickle_file(data=word_class_dict, filename=WORD_CLASS_DICT_FILENAME)
            data_store.upload_file("/tmp/" + MODEL_FILENAME+".index", MODEL_FILENAME+".index")
            data_store.upload_file("/tmp/" + MODEL_FILENAME+".meta", MODEL_FILENAME+".meta")
            data_store.upload_file("/tmp/" + MODEL_FILENAME+".data-00000-of-00001", MODEL_FILENAME+".data-00000-of-00001")
            data_store.upload_file("/tmp/" + WORD_CLASS_DICT_FILENAME, WORD_CLASS_DICT_FILENAME)
        return None

    @classmethod
    def load(cls, data_store):

        if type(data_store) is LocalFileSystem:
            word_class_dict = data_store.read_pickle_file(filename=WORD_CLASS_DICT_FILENAME)
            net = tflearn.input_data(shape=[None, int(word_class_dict["num_input"])])
            net = tflearn.fully_connected(net, 8)
            net = tflearn.fully_connected(net, 8)
            net = tflearn.fully_connected(net, int(word_class_dict["num_output"]), activation='softmax')
            net = tflearn.regression(net)
            model = tflearn.DNN(net)
            dl_model = data_store.read_dl_model(data=model, filename=MODEL_FILENAME)
        if type(data_store) is S3DataStore:
            data_store.download_file(MODEL_FILENAME+".index", "/tmp/" + MODEL_FILENAME+".index")
            data_store.download_file(MODEL_FILENAME+".meta", "/tmp/" + MODEL_FILENAME+".meta")
            data_store.download_file(MODEL_FILENAME+".data-00000-of-00001", "/tmp/" + MODEL_FILENAME+".data-00000-of-00001")
            data_store.download_file(WORD_CLASS_DICT_FILENAME, "/tmp/" + WORD_CLASS_DICT_FILENAME)
            temp_data_store = LocalFileSystem("/tmp/")
            word_class_dict = temp_data_store.read_pickle_file(filename=WORD_CLASS_DICT_FILENAME)
            net = tflearn.input_data(shape=[None, int(word_class_dict["num_input"])])
            net = tflearn.fully_connected(net, 8)
            net = tflearn.fully_connected(net, 8)
            net = tflearn.fully_connected(net, int(word_class_dict["num_output"]), activation='softmax')
            net = tflearn.regression(net)
            model = tflearn.DNN(net)
            dl_model = temp_data_store.read_dl_model(data=model, filename=MODEL_FILENAME)
        return ChatbotModel(words=word_class_dict["words"], classes=word_class_dict["classes"],
                            num_input=word_class_dict["num_input"], num_output=word_class_dict["num_output"],
                            dl_model=dl_model,response=word_class_dict["response"])

    @classmethod
    def preprocess_training_data(cls, intents):
        words = []
        classes = []
        documents = []
        ignore_words = ['?']
        response = {}
        for intent in intents['intents']:
            for pattern in intent['patterns']:
                if pattern is None:
                    pattern = "nothing specified"
                w = nltk.word_tokenize(pattern)
                words.extend(w)
                documents.append((w, intent['tag']))
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])
            response[intent["tag"]] = intent["responses"]

        stemmer = LancasterStemmer()

        words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))

        print(len(documents), "documents")
        print(len(classes), "classes", classes)
        print(len(words), "unique stemmed words", words)

        training = []
        output_empty = [0] * len(classes)

        for doc in documents:
            bag = []
            pattern_words = doc[0]
            pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
            for w in words:
                bag.append(1) if w in pattern_words else bag.append(0)
            output_row = list(output_empty)
            output_row[classes.index(doc[1])] = 1

            training.append([bag, output_row])
        random.shuffle(training)
        training = np.array(training)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        return train_x, train_y,words,classes,response

    @classmethod
    def train(cls, data_store):
        intents = data_store.read_json_file(filename=INTENT_FILENAME)
        train_X,train_Y,words,classes,response = cls.preprocess_training_data(intents)

        tf.reset_default_graph()
        net = tflearn.input_data(shape=[None, len(train_X[0])])
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, 8)
        net = tflearn.fully_connected(net, len(train_Y[0]), activation='softmax')
        net = tflearn.regression(net)

        model = tflearn.DNN(net)
        model.fit(train_X, train_Y, n_epoch=70, batch_size=8, show_metric=True)
        return ChatbotModel(words=words, classes=classes,
                            num_input=len(train_X[0]), num_output=len(train_Y[0]),
                            dl_model=model,response=response)

    def predict(self, sentence):
        results = self.dl_model.predict([bow(sentence, self.words)])[0]
        results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append((self.classes[r[0]], r[1]))
        return return_list
