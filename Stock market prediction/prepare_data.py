import nltk
import nltk.classify.util
import nltk.metrics
import pickle
import codecs

def freq_word(trainset, stopwords):
    '''

    :param trainset: pickle file trainset
    :return: the top1000 or top2000 freq words
    '''
    doc = []
    for (text, label) in trainset:
        for content in text:
            content = [words for words in content if words not in stopwords]
            doc.extend(content)
    freq_word = nltk.FreqDist(doc)
    word_features_1000 = list(freq_word)[:1000]
    pickle.dump(word_features_1000, open('word_features_1000.pkl', 'wb'))
    word_features_3000 = list(freq_word)[:3000]
    pickle.dump(word_features_3000, open('word_features_3000.pkl', 'wb'))
if __name__ == '__main__':
    f = codecs.open('./6_senti_data/stopword.txt', 'r', 'gbk')
    stopwords = [w.strip() for w in f.readlines()]
    f.close()
    stopwords.extend(['\n', '\t', ' '])

    trainset = pickle.load(open('trainset.pkl', 'rb'))
    print(trainset[0])
    freq_word(trainset, stopwords)