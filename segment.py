from __future__ import unicode_literals, division

import sys
import codecs
import argparse
import string
from collections import defaultdict, Counter

import numpy as np
import pickle
from itertools import combinations

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
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--vectors', '-v', type=argparse.FileType('rb'), default=sys.stdin,
        metavar='PATH',
        help="Serialzed dict of word vectors")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w+'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")

    return parser

def get_segmentations(word_in):
    word_segmentations = []
    cuts = []
    for i in range(0,len(word_in)):
        cuts.extend(combinations(range(1,len(word_in)),i))
    for i in cuts:
        last = 0
        output = []
        for j in i:
            output.append(word_in[last:j])
            last = j
        output.append(word_in[last:])
        word_segmentations.append(output)
    return word_segmentations

def get_vocabulary(fobj):
    """Read text and return dictionary that encodes vocabulary
    """
    fobj.seek(0)
    vocab = Counter()
    for line in fobj:
        for word in line.split():
            vocab[word] += 1
    return vocab


def get_similarity(word1, word2):
    if (word1 not in word_vectors) or (word2 not in word_vectors):
        return 0
    return np.dot(word_vectors[word1], word_vectors[word2])


#Given a list of two words in segmentation form, check if they share a common substring
def check_common_substring(seg1, seg2):
    for i in range(len(seg1) - 1):
        for j in range(len(seg2) - 1):
            if (seg1[i] == seg2[j]) and (seg1[i+1] == seg2[j+1]):
                return True
    return False


if __name__ == '__main__':
    #Main hyperparameter!
    gamma = 0

    parser = create_parser()
    args = parser.parse_args()
    word_vectors = pickle.load(args.vectors)
    segmentations = {}
    #Set up initial dict of segmentations...
    for line in args.input:
        contents = line.split()
        for word in contents:
            if word not in segmentations:
                segmentations[word] = list(word)

    vocab = get_vocabulary(args.input)
    
    #Core algorithm
    for word, freq in vocab.items():
        lowest_cost = float("inf")
        best_seg = None
        candidate_segs = get_segmentations(word)

        for seg in candidate_segs:
            cost = freq*len(seg)
            total_similarity = 0
            for other_word, other_seg in segmentations.items():
                if check_common_substring(seg, other_seg):
                    total_similarity += get_similarity(word, other_word)
            #Incorperates the similarity part of the cost function.
            cost -= gamma*total_similarity

            if cost < lowest_cost:
                lowest_cost = cost
                best_seg = seg

        segmentations[word] = best_seg

    #Write the word segmentations to the output file
    for word, final_seg in segmentations.items():
        delimited_seg = " ".join(final_seg)
        args.output.write(delimited_seg)
        args.output.write('\n')













        



