import nltk.metrics
import pickle
import time
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

def gen_features(sent, w): # w is the weight of title when w = 0, there is only content.
    features = {}
    n = len(sent) // 2
    word_features = pickle.load(open('word_features_3000.pkl', 'rb'))
    f = open('./6_senti_data/pos.txt')
    pos = f.readlines()
    f.close()
    f = open('./6_senti_data/neg.txt')
    neg = f.readlines()
    f.close()
    #global word_features
    for t in [2* i for i in range(n)]:
        for word in sent[t]:
            if word in word_features:  # 由于加入了title不好计算tf-idf中的idf,title和content不是一个量级
                features[word] = True
            if word in pos:
                features[word] = w  # 改进点
            elif word in neg:
                features[word] = -w
    for c in [2 * i + 1 for i in range(n)]: # 由于加入了title不好计算tf-idf中的idf,title和content不是一个量级
        for word in sent[c]:
            if word in word_features:  #
                features[word] = True
            if word in pos:
                features[word] = 1
            elif word in neg:
                features[word] = -1
    return features

def train_data(trainset, testset, w):
    training = [(gen_features(sent, w), label) for (sent, label) in trainset]
    testing = [(gen_features(sent, w), label) for (sent, label) in testset]

    return training, testing

if __name__ == '__main__':
    start = time.time()
    print('start')
    f = open('./6_senti_data/pos.txt')
    pos = f.readlines()
    f.close()
    # pos_dict = {}
    # for words in pos:
    #     word, score = words.strip().split(',')
    #     pos_dict[word] = float(score)
    # print('running here 1')
    f = open('./6_senti_data/neg.txt')
    neg = f.readlines()
    f.close()
    neg_dict = {}
    for words in neg:
        word, score = words.strip().split(',')
        neg_dict[word] = float(score)
    print('running here 2')
    # f =open('./6_senti_data/degree.txt')
    # degree = f.readlines()
    # f.close()
    # degree_dict = {}
    # for words in degree:
    #     word, score = words.strip().split(',')
    #     degree_dict[word] = float(score)
    #
    # f = open('./6_senti_data/no.txt')
    # no = f.readlines()
    # f.close()
    # no_dict = {}
    # for words in no:
    #     word = words.strip()
    #     no_dict[word] = float(-1)

    word_features = pickle.load(open('word_features_3000.pkl', 'rb'))
    trainset = pickle.load(open('trainset.pkl', 'rb'))
    testset = pickle.load(open('test.pkl', 'rb'))
    print('running here 3')
    w = 2
    training, testing = train_data(trainset, testset, w)
    pickle.dump(training, open('training.pkl', 'wb'))
    pickle.dump(testing, open('training.pkl', 'wb'))
    print('model1' + '*'*20)
    classifier = nltk.NaiveBayesClassifier.train(training)
    pickle.dump(classifier, open('./5_models/' + 'NaiveBayes' + '.pkl', 'wb'))

    print('model2' + '*' * 20)
    classifier = SklearnClassifier(BernoulliNB())
    classifier.train(training)
    pickle.dump(classifier, open('./5_models/' + 'BernoulliNB' + '.pkl', 'wb'))

    print('model3' + '*' * 20)
    classifier = SklearnClassifier(SVC())
    classifier.train(training)
    pickle.dump(classifier, open('./5_models/' + 'SVC' + '.pkl', 'wb'))

    print('model4' + '*' * 20)
    classifier = SklearnClassifier(LinearSVC())
    classifier.train(training)
    pickle.dump(classifier, open('./5_models/' + 'LinearSVC' + '.pkl', 'wb'))

    print('model5' + '*' * 20)
    classifier = SklearnClassifier(NuSVC())
    classifier.train(training)
    pickle.dump(classifier, open('./5_models/' + 'NuSVC' + '.pkl', 'wb'))

    print('model6' + '*' * 20)
    classifier = SklearnClassifier(LogisticRegression())
    classifier.train(training)
    pickle.dump(classifier, open('./5_models/' + 'LogisticRegression' + '.pkl', 'wb'))
    stop = time.time()
    print('time is:',str(stop - start))