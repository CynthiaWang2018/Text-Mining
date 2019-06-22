import numpy as np
import time
import ast
import nltk
from nltk.corpus import reuters as cp   # 1 corpus
from collections import deque
'''
hyperparameter
1.ngram = 0,1,2 unigram, bigram, trigram
'''
# PART 1
def preprocessing():
    '''
    preprocess the following dataset
    1. reuters or brown news corpus
    2. 'testdata.txt'
    3. 'vocab.txt'
    -----------------------------------------------------------------------
    preprocess things
    1.punctuation ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']
    -----------------------------------------------------------------------
    :return: 1. corpus_content(去掉标点的corpus), ngram_dict(ngram字典), N(corpus单词数, V(corpus去重单词数)
             2. testdata_list
             3. vocab_list
    '''
    corpus = cp.sents(categories=cp.categories())
    # The sents() function divides the text up into its sentences, where each sentence is a list of words
    corpus_content = []
    ngram_dict = {}
    for sents in corpus:
        sents = ['<s>'] + sents + ['</s>']
        for word in sents:
            if word in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']:
                sents.remove(word)
        corpus_content.extend(sents)
        for n in [1,2,3]: # unigram, bigram, trigram
            if len(sents) < n:
                continue
            else:
                for i in range(n, len(sents)+1):
                    gram = sents[i-n: i]
                    key = ' '.join(gram)
                    if key not in ngram_dict:
                        ngram_dict[key] = 1
                    else:
                        ngram_dict[key] += 1
    vocabulary = {}.fromkeys(corpus_content).keys()
    V = len(vocabulary)
    N = len(corpus_content)

    testdata_path = './testdata.txt'
    testdata_file = open(testdata_path,'r')
    testdata_list = []
    for line in testdata_file:
        line = line.split('\t')
        line[2] = nltk.word_tokenize(line[2])
        line[2] = ['<s>'] + line[2] + ['</s>']
        for word in line[2]:
            if word in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']:
                line[2].remove(word)
        testdata_list.append(line)

    vocab_path = './vocab.txt'
    vocab_file = open(vocab_path,'r')
    vocab_list = []
    for line in vocab_file:
        vocab_list.append(line[:-1])
    return corpus_content, ngram_dict, N, V, testdata_list, vocab_list
# PART 2
def language_model(ngram_dict, V, phrase, ngram, k, N):
    '''
    :param ngram_dict: the number of certain ngram
    :param V: the length of vocabulary
    :param phrase: input words or sentence
    :param ngram: unigram, bigram, trigram
    :param k: add k smoothing 1, 0.1, 0.01
    :param N: the number of words in corpus
    :return: log(pi)
    '''
    if ngram == 0:
        if phrase[0] in ngram_dict:
            pi = (ngram_dict[phrase[0]] + k)/(N + k*V)
        else:
            pi = k / (N + k*V)
        return np.log(pi)
    else:
        key1 = ' '.join(phrase) # ngram_dict[str] so need to change the list to string
        key2 = ' '.join(phrase[:-1])
        if key1 in ngram_dict and key2 in ngram_dict:
            pi = (ngram_dict[key1] + k)/(ngram_dict[key2] + k*V)
        elif key1 in ngram_dict:
            pi = (ngram_dict[key1] + k)/(k*V)
        elif key2 in ngram_dict:
            pi = k/(ngram_dict[key2] + k*V)
        else:
            pi = 1/V
        return np.log(pi)
# PART 3
END = '$'
def make_trie(vocab):
    trie = {}
    for word in vocab:
        t = trie
        for c in word:
            if c not in t: t[c] = {}
            t = t[c]
        t[END] = {}
    return trie

def get_candidate(trie, word, edit_distance=1):
    que = deque([(trie, word, '', edit_distance)])
    while que:
        trie, word, path, edit_distance = que.popleft()
        if word == '':
            if END in trie:
                yield path
            if edit_distance > 0:
                for k in trie:
                    if k != END:
                        que.appendleft((trie[k], '', path+k, edit_distance-1))
        else:
            if word[0] in trie:
                que.appendleft((trie[word[0]], word[1:], path+word[0], edit_distance))
            if edit_distance > 0:
                edit_distance -= 1
                for k in trie.keys() - {word[0], END}:
                    que.append((trie[k], word[1:], path+k, edit_distance))
                    que.append((trie[k], word, path+k, edit_distance))
                que.append((trie, word[1:], path, edit_distance))
                if len(word) > 1:
                    que.append((trie, word[1]+word[0]+word[2:], path, edit_distance))

def editType(candidate, word):
    edit = [False] * 4
    correct = ""
    error = ""
    x = ''
    w = ''
    for i in range(min([len(word), len(candidate)]) - 1):
        if candidate[0:i + 1] != word[0:i + 1]:
            if candidate[i:] == word[i - 1:]:
                edit[1] = True
                correct = candidate[i - 1]
                error = ''
                x = candidate[i - 2]
                w = candidate[i - 2] + candidate[i - 1]
                break
            elif candidate[i:] == word[i + 1:]:
                correct = ''
                error = word[i]
                if i == 0:
                    w = '#'
                    x = '#' + error
                else:
                    w = word[i - 1]
                    x = word[i - 1] + error
                edit[0] = True
                break
            if candidate[i + 1:] == word[i + 1:]:
                edit[2] = True
                correct = candidate[i]
                error = word[i]
                x = error
                w = correct
                break
            if candidate[i] == word[i + 1] and candidate[i + 2:] == word[i + 2:]:
                edit[3] = True
                correct = candidate[i] + candidate[i + 1]
                error = word[i] + word[i + 1]
                x = error
                w = correct
                break
    candidate = candidate[::-1]
    word = word[::-1]
    for i in range(min([len(word), len(candidate)]) - 1):
        if candidate[0:i + 1] != word[0:i + 1]:
            if candidate[i:] == word[i - 1:]:
                edit[1] = True
                correct = candidate[i - 1]
                error = ''
                x = candidate[i - 2]
                w = candidate[i - 2] + candidate[i - 1]
                break
            elif candidate[i:] == word[i + 1:]:

                correct = ''
                error = word[i]
                if i == 0:
                    w = '#'
                    x = '#' + error
                else:
                    w = word[i - 1]
                    x = word[i - 1] + error
                edit[0] = True
                break
            if candidate[i + 1:] == word[i + 1:]:
                edit[2] = True
                correct = candidate[i]
                error = word[i]
                x = error
                w = correct
                break
            if candidate[i] == word[i + 1] and candidate[i + 2:] == word[i + 2:]:
                edit[3] = True
                correct = candidate[i] + candidate[i + 1]
                error = word[i] + word[i + 1]
                x = error
                w = correct
                break
    if word == candidate:
        return "None", '', '', '', ''
    if edit[1]:
        return "Deletion", correct, error, x, w
    elif edit[0]:
        return "Insertion", correct, error, x, w
    elif edit[2]:
        return "Substitution", correct, error, x, w
    elif edit[3]:
        return "Reversal", correct, error, x, w

def loadConfusionMatrix():
    f=open('addconfusion.data', 'r')
    data=f.read()
    f.close
    addmatrix=ast.literal_eval(data)
    #--------------------------------
    f=open('subconfusion.data', 'r')
    data=f.read()
    f.close
    submatrix=ast.literal_eval(data)
    # --------------------------------
    f=open('revconfusion.data', 'r')
    data=f.read()
    f.close
    revmatrix=ast.literal_eval(data)
    # --------------------------------
    f=open('delconfusion.data', 'r')
    data=f.read()
    f.close
    delmatrix=ast.literal_eval(data)
    # --------------------------------
    return addmatrix, submatrix, revmatrix, delmatrix

def channelModel(x,y, edit, corpus):
    corpus2 = ' '.join(corpus)
    if edit == 'add':
        if x+y in addmatrix and corpus2.count(' '+y) and corpus2.count(x):
            if x == '#':
                return (addmatrix[x+y] + 1)/corpus2.count(' '+y)
            else:
                return (addmatrix[x+y] + 1)/corpus2.count(x)
        else:
            return 1 / len(corpus)
    if edit == 'sub':
        if (x+y)[0:2] in submatrix and corpus2.count(y):
            return (submatrix[(x+y)[0:2]] +1)/corpus2.count(y)
        elif (x+y)[0:2] in submatrix:
            return (submatrix[(x+y)[0:2]] +1)/len(corpus)
        elif corpus2.count(y):
            return 1/corpus2.count(y)
        else:
            return 1 / len(corpus)
    if edit == 'rev':
        if x+y in revmatrix and corpus2.count(x+y):
            return (revmatrix[x+y] + 1)/corpus2.count(x+y)
        elif x+y in revmatrix:
            return (revmatrix[x+y] + 1) / len(corpus)
        elif corpus2.count(x+y):
            return 1 / corpus2.count(x+y)
        else:
            return 1 / len(corpus)
    if edit == 'del':
        if x+y in delmatrix and corpus2.count(x+y):
            return (delmatrix[x+y] + 1)/corpus2.count(x+y)
        elif x+y in delmatrix:
            return (delmatrix[x+y] + 1)/len(corpus)
        elif corpus2.count(x+y):
            return 1/corpus2.count(x+y)
        else:
            return 1 / len(corpus)

# PART 4
def spell_correct(vocab, testdata, ngram_dict, corpus, V, trie, ngram, k):
    test_path = './testdata.txt'
    test_file = open(test_path, 'r')
    tmp = []
    for line in test_file:
        item = line.split('\t')
        del item[1]
        tmp.append('\t'.join(item))
    result_path = './result.txt'
    result_file = open(result_path, 'w')
    for item in testdata:
        for words in item[2][1:-1]: # 掠过最开始和最结束的<s>,</s>
            if (words in vocab):
                continue
            else:
                if (list(get_candidate(trie, words, edit_distance=1))):
                    candidate_list = list(get_candidate(trie, words, edit_distance=1))
                else:
                    candidate_list = list(get_candidate(trie, words, edit_distance=2))
                candidate_pi = []
                for candidate in candidate_list:
                    if(ngram == 0):
                        candidate_pi.append(
                            language_model(ngram_dict, V, [candidate], ngram, k, N))  # 0 = unigram, 1 = bigram
                    else:
                        edit = editType(candidate, words)
                        if edit == None:
                            candidate_pi.append(
                                language_model(ngram_dict, V, [candidate], ngram, k, N))
                            continue
                        if edit[0] == "Insertion":
                            channel_p = np.log(channelModel(edit[3][0], edit[3][1], 'add', corpus))
                        if edit[0] == 'Deletion':
                            channel_p = np.log(channelModel(edit[4][0], edit[4][1], 'del', corpus))
                        if edit[0] == 'Reversal':
                            channel_p = np.log(channelModel(edit[4][0], edit[4][1], 'rev', corpus))
                        if edit[0] == 'Substitution':
                            channel_p = np.log(channelModel(edit[3], edit[4], 'sub', corpus))
                        word_index = item[2][1:-1].index(words)
                        pre_phrase = item[2][1:-1][(word_index - ngram): word_index] + [candidate]
                        post_phrase = [candidate] + item[2][1:-1][(word_index + 1): word_index + ngram + 1]
                        p = language_model(ngram_dict, V, pre_phrase, ngram, k, N) + \
                            language_model(ngram_dict, V, post_phrase, ngram, k, N)
                        p = p + channel_p
                        candidate_pi.append(p)
                index = candidate_pi.index(max(candidate_pi))
                tmp[int(item[0]) - 1] = tmp[int(item[0]) - 1].replace(words, candidate_list[index])
        result_file.write(tmp[int(item[0]) - 1])

if __name__ == '__main__':
    start = time.time()
    print('PART1 Preprocessing Dataset')
    corpus_content, ngram_dict, N, V, testdata, vocab = preprocessing()
    addmatrix, submatrix, revmatrix, delmatrix = loadConfusionMatrix()
    trie = make_trie(vocab)
    print('PART2 Spell Correction')
    k = 0.01  # add-k smoothing
    spell_correct(vocab, testdata, ngram_dict, corpus_content, V, trie, 1, k) # 1:bigram
    #spell_correct(vocab, testdata, ngram_dict, corpus_content, V, trie, 0, k) # unigram
    #spell_correct(vocab, testdata, ngram_dict, corpus_content, V, trie, 2, k) # trigram
    stop = time.time()
    print('This program time is ' + str(stop - start) + '\n')
# accuaracy :85.9%



