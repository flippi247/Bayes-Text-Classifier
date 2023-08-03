import nltk
from collections import defaultdict

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')


class TextProcessor:

    def __init__(self, text):
        self.stopword_list = stopwords.words('english')
        self.raw_text = text

        self.text_as_dict = None
        self.classes = []
        self.texts = []

    def convert_text_to_dictionary(self):
        texts = self.raw_text.split('\n')
        d = defaultdict(list)
        for text in texts:
            splitted_text = text.split('\t')
            if len(splitted_text) == 2:
                self.texts.append(self.create_text_list_with_seperate_words(splitted_text[1]))
                d[splitted_text[0]].append(self.create_text_list_with_seperate_words(splitted_text[1]))
                self.classes.append(splitted_text[0])
        self.text_as_dict = d

    def create_text_list_with_seperate_words(self, text):
        l = []
        words = text.split(" ")
        for word in words:
            l.append(word)
        return l

    def remove_stopwords(self):
        for i in range(len(self.texts)):
            for word in self.texts[i]:
                if word in self.stopword_list:
                    self.texts[i].remove(word)
        if self.text_as_dict is not None:
            for k, v in self.text_as_dict.items():
                for wordlist in self.text_as_dict[k]:
                    for word in wordlist:
                        if word in self.stopword_list:
                            wordlist.remove(word)

    def stemm_text(self):
        stemmer = PorterStemmer()
        l = []
        for i in range(len(self.texts)):
            l.append([])
            for word in self.texts[i]:
                l[i].append(stemmer.stem(word))
        self.texts = l
        if self.text_as_dict is not None:
            for i in self.text_as_dict.keys():
                l2 = []
                for j in self.text_as_dict[i]:
                    for word in j:
                        l2.append(stemmer.stem(word))
                self.text_as_dict[i] = l2

    def lemm_text(self):
        lemmer = WordNetLemmatizer()
        l = []
        for i in range(len(self.texts)):
            l.append([])
            for word in self.texts[i]:
                l[i].append(lemmer.lemmatize(word))
        self.texts = l
        if self.text_as_dict is not None:
            for i in self.text_as_dict.keys():
                l2 = []
                for j in self.text_as_dict[i]:
                    for word in j:
                        l2.append(lemmer.lemmatize(word))
                self.text_as_dict[i] = l2