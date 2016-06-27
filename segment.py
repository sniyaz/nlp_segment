from __future__ import unicode_literals, division

import sys
import codecs
import argparse
import string
from collections import defaultdict, Counter

import numpy as np
import pickle
from itertools import combinations
from random import shuffle

import pdb

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
    """Set up our data structures...
    """
    fobj.seek(0)
    vocab = Counter()
    for line in fobj:
        for word in line.split():
            #HACK since we don't have the similarity net yet...
            if word not in word_vectors:
                continue
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


def get_next_seg(candidate_segs):
    best_seg = None
    original_word = "".join(candidate_segs[0])
    lowest_cost = float("inf")
    #Memoization table that maps segments to their score. Intractability is bad.
    memoized = {}

    for seg in candidate_segs:
        cost = vocab[original_word]*len(seg)
        func_seg = 0;
        for part in seg:
            if part in memoized:
                func_seg += memoized[part]
            elif part not in quick_find:
                memoized[part] = 0
            else:
                shared = quick_find[part]
                func_part = 0
                for other in shared:
                    func_part += get_similarity(original_word, other[0])
                func_seg += func_part
                memoized[part] = func_part

        #Incorperates the similarity part of the cost function.
        cost -= gamma*func_seg

        if cost < lowest_cost: 
            lowest_cost = cost
            best_seg = seg
    
    return best_seg

            

if __name__ == '__main__':
    #Main hyperparameter!
    gamma = 0.01

    parser = create_parser()
    args = parser.parse_args()
    word_vectors = pickle.load(args.vectors)
    segmentations = {}
    
    #Dict that maps characters to words containing them (for speed)
    quick_find = {}
    vocab = get_vocabulary(args.input)



    #Each words starts totally segmented..
    #By the way, I'm keeping index information in the quick find dict just in case...
    for word in vocab:
        segmentations[word] = list(word)
        for idx, c in enumerate(word):
            if c not in quick_find:
                quick_find[c] = set([(word, idx)])
            else:
                quick_find[c].add((word, idx))

    print("SIZE QUICK FIND")
    print(len(quick_find.keys()))

    print("SIZE VOCAB")
    print(len(vocab.keys()))
    
    i = 0
    #Core algorithm
    #Traverse words in random order
    word_list = [word for word in vocab]
    shuffle(word_list)
    for word in word_list:
        print(i)
        print(word)

        candidate_segs = get_segmentations(word)
        best_seg = get_next_seg(candidate_segs)

        segmentations[word] = best_seg
        #Update that quick find data_structure!
        for idx, c in enumerate(word):
            quick_find[c].remove((word, idx))

        for idx, part in enumerate(best_seg):
            if part not in quick_find:
                quick_find[part] = set([(word, idx)])
            else:
                part_set = quick_find[part]
                part_set.add((word, idx))

        i += 1

        
      
    #Write the word segmentations to the output file
    for word in word_list:
        final_seg = segmentations[word]
        delimited_seg = " ".join(final_seg)
        args.output.write(delimited_seg)
        args.output.write('\n')













        



