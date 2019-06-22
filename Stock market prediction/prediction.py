from tqdm import tqdm # a tool for progress bar
import time
import heapq # heapq.nlargest(num, list) 前几大的元素
import codecs
import math
import jieba
import nltk
import re
from collections import Counter
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

jieba.load_userdict('./2_add_jieba/financeword.txt')
jieba.load_userdict('./2_add_jieba/financeword2.txt')

def find_news(news_id):
    '''
    :param news_id: the list in train or test.e.g.[44374, 44416, 2913537, 2913541]
    :return: combined correspond news
    '''
    index_list = [id_list.index(x) for x in news_id]
    return [news[x] for x in index_list]
# print(find_news([44374, 44416, 2913537, 2913541]))

def Sent2Word(sentence):
    '''
    :param sentence: one sentence
    :return: segment words
    '''
    seg = list(jieba.cut(sentence))
    seg_after = []
    for word in seg:
        if word not in neg_words: # fliter the stopwords + stock + my_word
            seg_after.append(word)
    seg_after = [x for x in seg_after if not re.findall('\d', x) and len(x) > 1] # filter all the number and left the length > 1
    return seg_after

def word_freq(corpus):
    word_freq_n = Counter()
    for i in corpus:
        word_freq_n.update(set(i))  # set :how many doc does the word occurrence
        #word_freq_n.update(i)
    return word_freq_n

def idf(word_freq_n, N):
    '''
    calculate the idf value
    :param word_freq_n: times of word occurrence
    :param N: number of doc
    :return: idf prob value
    '''
    # tmp = math.log(N + 1) +1
    idf_prob = {}
    for key, value in word_freq_n.items():
        idf_prob[key] = math.log((N + 1) / (value + 1)) + 1
    return idf_prob

def nlarge_tfidf_words(corpus, idf_prob, n=300):
    '''
    get the words which have large tfidf value
    :param corpus: word list
    :param idf_prob:  idf_prob
    :param n: choose n
    :return: n tf-idf words
    '''
    words = []
    combine_corpus = []
    for i in corpus:
        combine_corpus += i
    counter_combine_corpus = Counter(combine_corpus)
    tmp_dict = {}
    for key, value in counter_combine_corpus.items():
        tmp_dict[key] = value * idf_prob[key]
    return heapq.nlargest(n, tmp_dict, key=tmp_dict.get)

def gen_features(words_list, tfidf_features):
    features = {}
    for word in tfidf_features:
        features['{}'.format(word)] = word in words_list
    return features

if __name__ == '__main__':
    start = time.time()
    # ------------------------------------------------------------------------------------------------------------------
    print('Step 1:reading data')
    f = codecs.open('./1_raw_dataset/news.txt', 'r', 'utf8')
    news = [eval(i) for i in f.readlines()]
    f.close()
    print(news[0])

    f = open('./1_raw_dataset/train.txt', 'r')
    train = [i.split() for i in f.readlines()]
    f.close()
    for i in range(len(train)):
        train[i][1] = train[i][1].split(',')
        train[i][1] = list(map(lambda x: int(x), train[i][1]))
        train[i].reverse()
    print(train[0])

    f = open('./1_raw_dataset/test.txt', 'r')
    test = [i.split() for i in f.readlines()]
    f.close()
    for i in range(len(test)):
        test[i][1] = test[i][1].split(',')
        test[i][1] = list(map(lambda x: int(x), test[i][1]))
        test[i].reverse()
    print(test[0])
    # ------------------------------------------------------------------------------------------------------------------
    print('Step 2: reading stopwords')
    f = codecs.open('./3_stopword/stopword.txt', 'r', 'gbk')
    stopwords = [x.strip() for x in f.readlines()]
    f.close()
    print(stopwords[0])
    # ------------------------------------------------------------------------------------------------------------------
    neg_words = stopwords # + stock
    id_list = []
    for new in news:
        id_list.append(new['id'])
    all_id_train = [x[0] for x in train]
    # print(all_id_train[0])
    train_corpus = []
    # i = [44374, 44416, 2913537, 2913541]
    # print(find_news(i))
    # print(find_news([44374, 44416, 2913537, 2913541]))
    for i in tqdm(all_id_train):
        # print(i)
        # print(type(i))
        # print('start find news')
        # print(find_news([44374, 44416, 2913537, 2913541]))
        # find_news([44374, 44416, 2913537, 2913541])
        # news = find_news(i)
        x_news = find_news(i)  # don't use same variable name  !!
        # print('start find content')
        x_news = [x['content'] for x in x_news]
        news2word = [Sent2Word(x) for x in x_news]
        news_combine = []
        for j in news2word:
            news_combine += j  # combine the news content of one sample
        train_corpus.append(news_combine)
    N = len(train_corpus)
    print('N is ', N)
    # print(train_corpus[0])

    all_id_test = [x[0] for x in test]
    # print(all_id_test[0])
    test_corpus = []
    for i in tqdm(all_id_test):
        x_news = find_news(i)  # don't use same variable name  !!
        # print('start find content')
        x_news = [x['content'] for x in x_news]
        news2word = [Sent2Word(x) for x in x_news]
        news_combine = []
        for j in news2word:
            news_combine += j  # combine the news content of one sample
        test_corpus.append(news_combine)
    N = len(test_corpus)
    print('N is ', N)
    # print(test_corpus[0])
    # -------------------------------------------------------------------------------------------------------------------
    print('Step 3: generate features')
    word_freq_n = word_freq(train_corpus)
    idf_prob = idf(word_freq_n, N)
    tfidf_features = nlarge_tfidf_words(train_corpus, idf_prob, n=3000)
    print(train_corpus[0])
    print(gen_features(train_corpus[0], tfidf_features))
    train_data = []
    for i in tqdm(range(len(train_corpus))):
        train_data.append((gen_features(train_corpus[i], tfidf_features), train[i][1]))
    print(train_data[0])
    test_data = []
    for i in tqdm(range(len(test_corpus))):
        test_data.append((gen_features(test_corpus[i], tfidf_features), test[i][1]))
    print(test_data[0])
    # -------------------------------------------------------------------------------------------------------------------
    # print('Step 4: Train classifier')
    # naive = nltk.NaiveBayesClassifier.train(train_data)
    # print(nltk.classify.accuracy(naive, test_data))
    # result_path = './4_result/naiveBayes.txt'
    # result_file = open(result_path, 'w')
    # for item in test_data:
    #     result_file.write(naive.classify(item[0]) + '\n')
    # result_file.close()
    # print(naive.show_most_informative_features(5))
    #
    # print('*'*50)
    # classifier = SklearnClassifier(LogisticRegression())
    # print('Training LR...')
    # classifier.train(train_data)
    # result_path = './4_result/LR.txt'
    # result_file = open(result_path, 'w')
    # for item in test_data:
    #     result_file.write(classifier.classify(item[0]) + '\n')
    # result_file.close()
    #
    # print('*' * 50)
    # classifier = SklearnClassifier(BernoulliNB())
    # print('Training BernoulliNB...')
    # classifier.train(train_data)
    # result_path = './4_result/BernoulliNB.txt'
    # result_file = open(result_path, 'w')
    # for item in test_data:
    #     result_file.write(classifier.classify(item[0]) + '\n')
    # result_file.close()
    #
    # print('*' * 50)
    # classifier = SklearnClassifier(LinearSVC())
    # print('Training LinearSVC...')
    # classifier.train(train_data)
    # result_path = './4_result/LinearSVC.txt'
    # result_file = open(result_path, 'w')
    # for item in test_data:
    #     result_file.write(classifier.classify(item[0]) + '\n')
    # result_file.close()
    #
    # print('*' * 50)
    # classifier = SklearnClassifier(SVC())
    # print('Training SVC...')
    # classifier.train(train_data)
    # result_path = './4_result/SVC.txt'
    # result_file = open(result_path, 'w')
    # for item in test_data:
    #     result_file.write(classifier.classify(item[0]) + '\n')
    # result_file.close()
    #
    # print('*' * 50)
    # classifier = SklearnClassifier(NuSVC())
    # print('Training NuSVC...')
    # classifier.train(train_data)
    # result_path = './4_result/NuSVC.txt'
    # result_file = open(result_path, 'w')
    # for item in test_data:
    #     result_file.write(classifier.classify(item[0]) + '\n')
    # result_file.close()
    #
    # stop = time.time()
    # print('all time:', str(stop - start) + '\n')
    print('Step 4: Train classifier')
    naive = nltk.NaiveBayesClassifier.train(train_data)
    print(nltk.classify.accuracy(naive, test_data))
    result_path = './8_result3/naiveBayes.txt'
    result_file = open(result_path, 'w')
    for item in test_data:
        result_file.write(naive.classify(item[0]) + '\n')
    result_file.close()
    print(naive.show_most_informative_features(5))

    print('*' * 50)
    classifier = SklearnClassifier(LogisticRegression())
    print('Training LR...')
    classifier.train(train_data)
    result_path = './8_result3/LR.txt'
    result_file = open(result_path, 'w')
    for item in test_data:
        result_file.write(classifier.classify(item[0]) + '\n')
    result_file.close()

    print('*' * 50)
    classifier = SklearnClassifier(BernoulliNB())
    print('Training BernoulliNB...')
    classifier.train(train_data)
    result_path = './8_result3/BernoulliNB.txt'
    result_file = open(result_path, 'w')
    for item in test_data:
        result_file.write(classifier.classify(item[0]) + '\n')
    result_file.close()

    print('*' * 50)
    classifier = SklearnClassifier(LinearSVC())
    print('Training LinearSVC...')
    classifier.train(train_data)
    result_path = './8_result3/LinearSVC.txt'
    result_file = open(result_path, 'w')
    for item in test_data:
        result_file.write(classifier.classify(item[0]) + '\n')
    result_file.close()

    print('*' * 50)
    classifier = SklearnClassifier(SVC())
    print('Training SVC...')
    classifier.train(train_data)
    result_path = './8_result3/SVC.txt'
    result_file = open(result_path, 'w')
    for item in test_data:
        result_file.write(classifier.classify(item[0]) + '\n')
    result_file.close()

    print('*' * 50)
    classifier = SklearnClassifier(NuSVC())
    print('Training NuSVC...')
    classifier.train(train_data)
    result_path = './8_result3/NuSVC.txt'
    result_file = open(result_path, 'w')
    for item in test_data:
        result_file.write(classifier.classify(item[0]) + '\n')
    result_file.close()

    stop = time.time()
    print('all time:', str(stop - start) + '\n')


