'''
--data
    --'degree.txt': 程度词汇
    --'neg.txt': 消极词汇
    --'no.txt':否定词汇
    --'pos.txt':积极词汇
    --'stopword.txt':停用词
    --'financewoed.txt':金融词汇
    --'financeword2.txt':金融词汇

This python file obtains two '.pkl' files:'trainset.pkl','testset.pkl'
[[title1],[content1],[title2],[content2],...'-1']
'''
import codecs
import jieba
import pickle
from tqdm import tqdm

jieba.load_userdict('./2_add_jieba/financeword.txt')
jieba.load_userdict('./2_add_jieba/financeword2.txt')

def Sent2Word(sentence, stopwords):
    words = jieba.cut(sentence)
    seg = [x for x in words if x not in stopwords]

    return seg

def pickle_data(news, train, test):
    '''
    :param news:[{'id': 30819,
            'title': '航天科技集团深化军工改制 军工装备股再受青睐',
            'content': '据《中国航天报》报道，....'},{},...]
    :param train:[['+1','44374,44416,2913537,2913541'],[],...]
    :param test: the same as train
    :return:trainset:[[title1],[content1],[title2],[content2],...],'+1'
            testset:[[title1],[content1],[title2],[content2],...],'+1'
    '''
    # all_id = []
    # for new in news:
    #     all_id.append(new['id'])
    # #print(all_id)
    # print(all_id.index(44374))
    all_id = [x['id'] for x in news]
    trainset = []
    for i in tqdm(range(len(train))):
        id_list = train[i][1].split(',')
        id_list = list(map(lambda x: int(x), id_list))
        train_des = []
        for id in id_list:
            index = all_id.index(id)
            train_des.append(Sent2Word(news[index]['title'], stopwords))
            train_des.append(Sent2Word(news[index]['content'], stopwords))
        trainset.extend([(train_des, train[i][0])])
    print(trainset[0])
    pickle.dump(trainset, open('trainset.pkl', 'wb'))

    testset = []
    for i in tqdm(range(len(test))):
        id_list = test[i][1].split(',')
        id_list = list(map(lambda x: int(x), id_list))
        test_des = []
        for id in id_list:
            index = all_id.index(id)
            test_des.append(Sent2Word(news[index]['title'], stopwords))
            test_des.append(Sent2Word(news[index]['content'], stopwords))
        testset.extend([(test_des, train[i][0])])
    pickle.dump(testset, open('test.pkl', 'wb'))  #  'testset.pkl' will be better

if __name__ == '__main__':
    f = codecs.open('./1_raw_dataset/news.txt', 'r', 'utf8')
    news = [eval(i) for i in f.readlines()]
    f.close()

    f = codecs.open('./1_raw_dataset/train.txt', 'r')
    train = [i.split() for i in f.readlines()]
    f.close()

    f = codecs.open('./1_raw_dataset/test.txt', 'r')
    test = [i.split() for i in f.readlines()]
    f.close()

    f = codecs.open('./6_senti_data/stopword.txt', 'r', 'gbk')
    stopwords = [w.strip() for w in f.readlines()]
    f.close()
    stopwords.extend(['\n', '\t', ' '])
    pickle_data(news, train, test)