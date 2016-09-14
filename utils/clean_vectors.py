from __future__ import unicode_literals, division

import sys
import codecs
import argparse
import string
from collections import defaultdict, Counter
from spell import correct_word

sys.path.append("../evaluation/")
from evaluate_seg import get_gs_data

import numpy as np
import pickle

import pdb

sys.path.append("../")
from bpe import get_vocabulary_freq_table, get_mean

# hack for python2/3 compatibility
from io import open
argparse.open = open

# python 2/3 compatibility
if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="use Norvig spell checker to clean word_vectors")
    parser.add_argument(
        '--input', '-i', type=argparse.FileType('rb'), default=sys.stdin,
        metavar='PATH',
        help="Input vectors")
    parser.add_argument(
        '--corpus', '-c', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Corpus to do the cleaning over")
    parser.add_argument(
        '--gold_standard', '-gs', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Gold standard to do the cleaning over")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('wb'), default=sys.stdout,
        metavar='PATH',
        help="Output file for cleaned vectors")
    
    return parser

if __name__ == '__main__':
         
    parser = create_parser()
    args = parser.parse_args()

    word_vectors = pickle.load(args.input)
    in_vector_table = list(word_vectors.keys())
    in_mean = get_mean(set([(word,) for word in in_vector_table]), word_vectors)

    corpus_vocab = get_vocabulary_freq_table(args.corpus, word_vectors)
    vector_vocab = Counter()
    gold_standard = {}
    get_gs_data(args.gold_standard, gold_standard, []) 
    for word in word_vectors:
        vector_vocab[word] = corpus_vocab[word]

    missed = 0
    for word in list(corpus_vocab.keys()) + list(gold_standard.keys()):
        if word not in word_vectors:
            word_correction = correct_word(word, vector_vocab)
            if word_correction in word_vectors:
                word_vectors[word] = word_vectors[word_correction]
            else:
                word_vectors[word] = in_mean
                missed += 1

    
    pdb.set_trace()
    pickle.dump(word_vectors, args.output)




