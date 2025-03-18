import numpy as np
import pickle


def load_sentic_word():
    """
    load senticNet
    """
    path = './senticnet_word.txt'
    senticNet = {}
    fp = open(path, 'r')
    for line in fp:
        line = line.strip()
        if not line:
            continue
        word, sentic = line.split('\t')
        senticNet[word] = sentic
    fp.close()
    return senticNet

def dependency_adj_matrix(text, aspect, senticNet):
    word_list = text.split()
    seq_len = len(word_list)
    matrix = np.zeros((seq_len, seq_len)).astype('float32')

    for i in range(seq_len):
        word = word_list[i]
        if word in senticNet:
            sentic = float(senticNet[word]) + 1.0
        else:
            sentic = 0
        if word in aspect:
            sentic += 1.0
        for j in range(seq_len):
            matrix[i][j] += sentic
            matrix[j][i] += sentic
    for i in range(seq_len):
        if matrix[i][i] == 0:
            matrix[i][i] = 1

    return matrix


def process(filename):
    senticNet = load_sentic_word()
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')


if __name__ == '__main__':
    process('a')
