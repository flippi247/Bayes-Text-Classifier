from Naive_Bayes_Classifier import *
from Text_Processor import *
import urllib.request

# Read Train and Test Text
with urllib.request.urlopen('https://raw.githubusercontent.com/nikitasrivatsan/word2vec/master/data/20ng-train-all'
                            '-terms.txt') as f:
    train_text = f.read().decode('utf-8')
with urllib.request.urlopen('https://raw.githubusercontent.com/nikitasrivatsan/word2vec/master/data/20ng-test-all'
                            '-terms.txt') as f:
    test_text = f.read().decode('utf-8')

### Tests in File Test_Models delivered the best fitting Model below ###
train_data_processor = TextProcessor(train_text)
train_data_processor.convert_text_to_dictionary()
train_data_processor.remove_stopwords()
train_data_processor.stemm_text()

test_data_processor = TextProcessor(test_text)
test_data_processor.convert_text_to_dictionary()
test_data_processor.remove_stopwords()
test_data_processor.stemm_text()
print(test_data_processor.texts[2])
nb_clf = NaiveBayes()
nb_clf.fit(train_data_processor)
print(nb_clf.predict(test_data_processor.texts[2]))

print(test_data_processor.classes[2])
for k, v in nb_clf.predicted_class_likelihoods.items():
    print(f'Key: {k} Value: {v}')
for k, v in nb_clf.word_likelihoods_given_class.items():
    print(f'Klasse: {k}')
    for k2, v2 in v.items():
        if v2 == 0:
            print(f'Wort: {k2}\tWkeit: {v2} ')
