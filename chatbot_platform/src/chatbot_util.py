import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import time


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    stemmer = LancasterStemmer()
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words, show_details=False):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return np.array(bag)

def get_unique_number():
    current_milli_time = int(round(time.time() * 1000))
    return current_milli_time
