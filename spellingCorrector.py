import numpy as np
import ast
import nltk
from nltk.corpus import brown
from jiwer import wer, cer

printpath = './print.txt'
printfile = open(printpath, 'w')

# preprocessing
def preprocessing(genres):
    # store the vocabulary to a list
    vocabpath = './vocab.txt'
    vocabfile = open(vocabpath, 'r')
    vocab_list = []
    for line in vocabfile:
        vocab_list.append(line[:-1])  # subtract the '\n'

    # read testdata and preprocessing it, store it to a list
    testpath = './testdata.txt'
    testfile = open(testpath, 'r')

    testdata = []
    for paragraph in testfile:
        sentences = nltk.sent_tokenize(paragraph)

        # preprocessing sentence
        for sentence in sentences:
            sentence = nltk.word_tokenize(sentence)
            sentence = ['<s>'] + sentence + ['</s>']

            # remove string.punctuation
            for words in sentence[::]:  # use [::] to remove the continuous ';' ';'
                if (words in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']):  
                    sentence.remove(words)
                    continue

            testdata.append(sentence)

    # preprocessing the corpus
    corpus_sentences = list(brown.sents(categories=genres))

    corpus_text = []
    vocab_corpus = []
    for sentence in corpus_sentences:

        # remove string.punctuation
        for words in sentence[::]:  # use [::] to remove the continuous ';' ';'
            if (words in ['\'\'', '``', ',', '--', ';', ':', '(', ')', '&', '\'', '!', '?', '.']):  
                sentence.remove(words)
        sentence = [words.lower() for words in sentence]
        corpus_text.append(sentence)
        vocab_corpus.extend(sentence)

    V = len(set(vocab_corpus))
    return vocab_list, testdata, corpus_text, V

def count_N_grams(corpus, N):

    N_grams = {}

    for sentence in corpus:
        if N > 1:
          sentence = ['<s>']*(N-1) + sentence + ['</s>']
        else:
          sentence = ['<s>']*N + sentence + ['</s>']
        sentence = tuple(sentence)

        for i in range(len(sentence)-N+1):
            N_gram = sentence[i:i+N]
            if N_gram in N_grams.keys():
                N_grams[N_gram] += 1
            else:
                N_grams[N_gram] = 1
    return N_grams

def estimate_probability(word, previous_n_minus1_gram,
                         n_minus1_gram_counts, n_gram_counts):
    previous_n_minus1_gram = tuple(previous_n_minus1_gram)

    if (previous_n_minus1_gram+(word,)) in n_gram_counts.keys() and previous_n_minus1_gram in n_minus1_gram_counts.keys():
        numerator = n_gram_counts[previous_n_minus1_gram + (word,)]
        denominator = n_minus1_gram_counts[(previous_n_minus1_gram)]
        probability = numerator / denominator
    else:
        probability = 0
    return probability

def estimate_addalpha_probability(word, previous_n_minus1_gram,
                         n_minus1_gram_counts, n_gram_counts, vocabulary_size, alpha):
    previous_n_minus1_gram = tuple(previous_n_minus1_gram)

    if (previous_n_minus1_gram + (word,)) in n_gram_counts.keys() and previous_n_minus1_gram in n_minus1_gram_counts.keys():
        numerator = n_gram_counts[previous_n_minus1_gram + (word,)] + alpha
        denominator = n_minus1_gram_counts[previous_n_minus1_gram] + alpha * vocabulary_size
        probability = numerator / denominator
    else:
        probability = 1/vocabulary_size
    return probability

def get_candidate(vocab, word):
    "All edits that are one edit away from `word`."
    candidate = {}
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    for L, R in splits:
        if R and L:
            #insertion
            if (L + R[1:]) in vocab and (L + R[1:]) not in candidate.keys():
                candidate[L + R[1:]] = ("Insertion", R[0], L[-1])
            # deletion
            for c in letters:
                if (L + c + R) in vocab and (L + c + R) not in candidate.keys():
                    candidate[L + c + R] = ("Deletion", L[-1], c)
        if R:
            #substitution
            for c in letters:
                if (L + c + R[1:]) in vocab and (L + c + R[1:]) not in candidate.keys():
                    candidate[L + c + R[1:]] = ("Substitution", R[0], c)
        # Transposition
        if len(R) > 1:
            if (L + R[1] + R[0] + R[2:]) in vocab and (L + R[1] + R[0] + R[2:]) not in candidate.keys():
                candidate[L + R[1] + R[0] + R[2:]] = ("Transposition", R[0], R[1])
        
    return candidate

# Method to load Confusion Matrix from external data file.
def loadConfusionMatrix():
    f=open('insconfusion.data', 'r')
    data=f.read()
    f.close
    insmatrix=ast.literal_eval(data)
    f=open('subconfusion.data', 'r')
    data=f.read()
    f.close
    submatrix=ast.literal_eval(data)
    f=open('transconfusion.data', 'r')
    data=f.read()
    f.close
    transmatrix=ast.literal_eval(data)
    f=open('delconfusion.data', 'r')
    data=f.read()
    f.close
    delmatrix=ast.literal_eval(data)
    return insmatrix, submatrix, transmatrix, delmatrix

# Method to calculate channel model probability for errors.
def channelModel(x,y, edit, corpus):
    corpus_str = ''
    for sentence in corpus:
        corpus_str.join(sentence)

    if edit == "ins":
        if x+y in insmatrix and corpus_str.count(' '+y) and corpus_str.count(x):
            if x == '#':
                return (insmatrix[x+y] + 1)/corpus_str.count(' '+y)
            else:
                return (insmatrix[x+y] + 1)/corpus_str.count(x)
        else:
            return 1 / len(corpus)
    if edit == "sub":
        if (x+y)[0:2] in submatrix and corpus_str.count(y):
            
            return (submatrix[(x+y)[0:2]] +1)/corpus_str.count(y)
        elif (x+y)[0:2] in submatrix:
            return (submatrix[(x+y)[0:2]] +1)/len(corpus)
        elif corpus_str.count(y):
            return 1/corpus_str.count(y)
        else:
            return 1 / len(corpus)
    if edit == "trans":
        if x+y in transmatrix and corpus_str.count(x+y):
            return (transmatrix[x+y] + 1)/corpus_str.count(x+y)
        elif x+y in transmatrix:
            return (transmatrix[x+y] + 1) / len(corpus)
        elif corpus_str.count(x+y):
            return 1 / corpus_str.count(x+y)
        else:
            return 1 / len(corpus)
    if edit == "del":
        if x+y in delmatrix and corpus_str.count(x+y):
            return (delmatrix[x+y] + 1)/corpus_str.count(x+y)
        elif x+y in delmatrix:
            return (delmatrix[x+y] + 1)/len(corpus)
        elif corpus_str.count(x+y):
            return 1/corpus_str.count(x+y)
        else:
            return 1 / len(corpus)

def spell_correct(vocab, testdata, corpus, V, alpha):
    testpath = './testdata.txt'
    testfile = open(testpath, 'r')
    data = []
    for paragraph in testfile:
        sentences = nltk.sent_tokenize(paragraph)
        sentences[-1] += ('\n')
        for i in range(len(sentences)):
            if i > 0:
                sentences[i] = ' ' + sentences[i]
        data.extend(sentences)

    for sentence in testdata:
        for words in sentence: 
            if (words in vocab):
                continue
            else:
                if get_candidate(vocab, words).keys():
                    candidate_list = get_candidate(vocab, words)
                    
                    for candidate in candidate_list.keys():
                        if candidate_list[candidate][0] == "Insertion":
                            channel_p = channelModel(candidate_list[candidate][1], candidate_list[candidate][2], "ins", corpus)
                        elif candidate_list[candidate][0] == 'Deletion':
                            channel_p = channelModel(candidate_list[candidate][1], candidate_list[candidate][2], "del", corpus)
                        elif candidate_list[candidate][0] == 'Tranposition':
                            channel_p = channelModel(candidate_list[candidate][1], candidate_list[candidate][2], "trans", corpus)
                        else:
                            channel_p = channelModel(candidate_list[candidate][1], candidate_list[candidate][2], "sub", corpus)

                        word_index = sentence.index(words)
                        if alpha:
                            prior_p = estimate_addalpha_probability(candidate, [sentence[word_index-1]], unigram_counts, bigram_counts, V, alpha)
                        else:
                            prior_p = estimate_probability(candidate, [sentence[word_index-1]], unigram_counts, bigram_counts)

                        p = np.log(prior_p) + np.log(channel_p) if prior_p > 0 else float('inf')
                        candidate_list[candidate]+= (p,) 
                    
                    max_p = float('-inf')
                    max_p_candidate = ''
                    for candidate in candidate_list.keys():
                        if candidate_list[candidate][3] > max_p:
                            max_p = candidate_list[candidate][3]
                            max_p_candidate = candidate
                    data[testdata.index(sentence)] = data[testdata.index(sentence)].replace(words, max_p_candidate)
        resultfile.write(data[testdata.index(sentence)])

if __name__ == '__main__':

    resultpath = './result.txt'
    resultfile = open(resultpath, 'w+')

    anspath='./answer.txt'
    answerfile = open(anspath, 'r')
    testpath = './testdata.txt'
    testfile = open(testpath,'r')
    genres = ['news', 'religion', 'hobbies', 'science_fiction', 'romance', 'humor']

    print('Doing preprocessing, computing things. Please wait...')
    vocab, testdata, corpus_text, V = preprocessing(genres)
    insmatrix, submatrix, transmatrix, delmatrix = loadConfusionMatrix()

    print('Doing Spell Correcting...')
    unigram_counts = count_N_grams(corpus_text, 1)
    bigram_counts = count_N_grams(corpus_text, 2)
    #trigram_counts = count_N_grams(corpus_text, 3)

    alpha = 1  # add-lambda smoothing
    spell_correct(vocab, testdata, corpus_text, V, None)

    refs = answerfile.readlines()
    resultfile.seek(0) # move cursor to the start of the file
    pred = resultfile.readlines()

    wer_score = wer(refs, pred)
    cer_score = cer(refs, pred)
    print('WER score: ', wer_score)
    print('CER score: ', cer_score)
