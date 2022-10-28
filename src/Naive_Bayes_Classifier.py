from Text_Processor import *
from copy import deepcopy

class NaiveBayes:

    def __init__(self):
        self.predicted_class_likelihoods = {}

        self.class_likelihoods = {}
        self.word_likelihoods_given_class = {}

        self.class_count_all = 0

        self.sum_total_words_in_class = {}
        self.sum_each_word_in_class = {}

    def fit(self, data):
        self.get_class_likelihoods(data)
        self.get_word_count(data)

    def get_class_likelihoods(self, data):
        class_count_all = 0
        class_count_dict = {}
        for i in data.text_as_dict.keys():
            class_count_dict[i] = 0
            for j in data.text_as_dict[i]:
                class_count_all += 1
                class_count_dict[i] += 1
        for k in class_count_dict.keys():
            class_count_dict[k] = class_count_dict[k] / class_count_all
        self.class_count_all = class_count_all
        self.class_likelihoods = class_count_dict

    def get_word_count(self, data):
        for key in data.text_as_dict.keys():
            self.sum_total_words_in_class[key] = len(data.text_as_dict[key])
            self.sum_each_word_in_class[key] = defaultdict(int)
            for word in data.text_as_dict[key]:
                self.sum_each_word_in_class.get(key)[word] += 1

    def predict(self, word_list, alpha=1):
        for key in self.class_likelihoods.keys():
            if len(word_list) > 0:
                p_word_given_class = 1
            else:
                p_word_given_class = 0
            p_y = self.class_likelihoods[key]
            self.laplace_smoothing(word_list, alpha)
            self.calculate_word_likelihoods_given_class()
            for word in word_list:
                p_word_given_class *= self.word_likelihoods_given_class.get(key).get(word)
            p_word_given_class *= p_y
            self.predicted_class_likelihoods[key] = p_word_given_class
        max = 0
        key_to_max = ""
        for k, v in self.predicted_class_likelihoods.items():
            if v > max:
                max = v
                key_to_max = k
        return key_to_max

    def laplace_smoothing(self, word_list, alpha):
        s = len(set(word_list))
        for key in self.sum_each_word_in_class.keys():
            ## Zähle auf alle Trainingswörter und Testwörter pro Klasse +1
            for word in word_list:
                self.sum_each_word_in_class.get(key)[word] += 1
            for word in self.sum_each_word_in_class.get(key).keys():
                self.sum_each_word_in_class.get(key)[word] += 1
            ## Für alle Klassen addiere die Anzahl an Trainingswörtern und Testwörtern
            d = len(self.sum_each_word_in_class[key].keys())
            self.sum_total_words_in_class[key] += (alpha * (d + s))

    def calculate_word_likelihoods_given_class(self):
        for key in self.sum_each_word_in_class.keys():
            total_number_words_in_class = self.sum_total_words_in_class[key]
            self.word_likelihoods_given_class[key] = defaultdict(float)
            for k, v in self.sum_each_word_in_class[key].items():
                self.word_likelihoods_given_class.get(key)[k] = v / total_number_words_in_class
