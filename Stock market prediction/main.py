'''
this file used the following files
--load_data : trainset, testset
--prepare_data: top 3000 freq words
--model : 6 tuned models
'''
import nltk.metrics
import time
import pickle
from model import *

if __name__ == '__main__':
    trainset = pickle.load(open('trainset.pkl', 'rb'))
    print('running here1')
    testset = pickle.load(open('test.pkl', 'rb'))
    print('running here2')
    training, testing = train_data(trainset, testset, 2)
    print('running here3')
    classifer = pickle.load(open('./5_models/NaiveBayes.pkl', 'rb'))
    print('running here4')
    # naive = nltk.NaiveBayesClassifier.train(train_data)
    print(nltk.classify.accuracy(classifer, testing))
    result_path = './7_result2/naiveBayes.txt'
    result_file = open(result_path, 'w')
    for item in testing:
        result_file.write(classifer.classify(item[0]) + '\n')
    result_file.close()
    print(classifer.show_most_informative_features(5))

    classifer = pickle.load(open('./5_models/LogisticRegression.pkl', 'rb'))
    print('*' * 50)
    # classifier = SklearnClassifier(LogisticRegression())
    # print('Training LR...')
    # classifier.train(train_data)
    result_path = './7_result2/LR.txt'
    result_file = open(result_path, 'w')
    for item in testing:
        result_file.write(classifer.classify(item[0]) + '\n')
    result_file.close()

    classifer = pickle.load(open('./5_models/BernoulliNB.pkl', 'rb'))
    print('*' * 50)
    # classifier = SklearnClassifier(BernoulliNB())
    # print('Training BernoulliNB...')
    # classifier.train(train_data)
    result_path = './7_result2/BernoulliNB.txt'
    result_file = open(result_path, 'w')
    for item in testing:
        result_file.write(classifer.classify(item[0]) + '\n')
    result_file.close()

    classifer = pickle.load(open('./5_models/LinearSVC.pkl', 'rb'))
    print('*' * 50)
    # classifier = SklearnClassifier(LinearSVC())
    # print('Training LinearSVC...')
    # classifier.train(train_data)
    result_path = './7_result2/LinearSVC.txt'
    result_file = open(result_path, 'w')
    for item in testing:
        result_file.write(classifer.classify(item[0]) + '\n')
    result_file.close()

    classifer = pickle.load(open('./5_models/SVC.pkl', 'rb'))
    print('*' * 50)
    # classifier = SklearnClassifier(SVC())
    # print('Training SVC...')
    # classifier.train(train_data)
    result_path = './7_result2/SVC.txt'
    result_file = open(result_path, 'w')
    for item in testing:
        result_file.write(classifer.classify(item[0]) + '\n')
    result_file.close()

    classifer = pickle.load(open('./5_models/NuSVC.pkl', 'rb'))
    print('*' * 50)
    # classifier = SklearnClassifier(NuSVC())
    # print('Training NuSVC...')
    # classifier.train(train_data)
    result_path = './7_result2/NuSVC.txt'
    result_file = open(result_path, 'w')
    for item in testing:
        result_file.write(classifer.classify(item[0]) + '\n')
    result_file.close()