from __future__ import unicode_literals, division

import sys
import codecs
import argparse
import string
from collections import defaultdict, Counter
from random import randint

import numpy as np
import pickle
from itertools import combinations
from random import shuffle
import copy

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
    return np.dot(word_vectors[word1], word_vectors[word2])


#Given a list of two words in segmentation form, check if they share a common substring
def check_common_substring(seg1, seg2):
    for i in range(len(seg1) - 1):
        for j in range(len(seg2) - 1):
            if (seg1[i] == seg2[j]) and (seg1[i+1] == seg2[j+1]):
                return True
    return False



#Is merging a pair worth it?
def get_pair_action(pair):
    involved_words = quick_pairs[pair]
    #We only calculate the change in cost for computational reasons
    cost_delta = 0
    for word, first_index, second_index in involved_words:
        #On a merge, the length goes down by one. Reflect the first term of the cost function
        cost_delta -= vocab[word]
        #"Undo" the contributions to the second part of the cost function
        for other_word, index in quick_find[pair[0]]:
            #Hack to prevent a third calculation from the new symbol
            if (other_word, index, index + 1) in involved_words:
                continue
            if (word, pair[0]) in memoized_scores:
                cost_delta += gamma*memoized_scores[(word, pair[0])]
            else:
                cost_delta += gamma*get_similarity(word, other_word)
                memoized_scores[(word, pair[0])] = get_similarity(word, other_word)
        for other_word, index in quick_find[pair[1]]:
            if (word, pair[1]) in memoized_scores:
                cost_delta += gamma*memoized_scores[(word, pair[1])]
            else:
                cost_delta += gamma*get_similarity(word, other_word)
                memoized_scores[(word, pair[1])] = get_similarity(word, other_word)
        
        
    return cost_delta < 0



def merge_update(pair):
    try:
        new_symbol = "".join(pair)
        involved_words = quick_pairs[pair]
        #Edge cases can have you change the set as you iterate over it!
        while involved_words:
            word, first_index, second_index = involved_words.pop()
            #Remove the mapping of the word and old symbols from the quick_find structure
            quick_find[pair[0]].remove((word, first_index))
            quick_find[pair[1]].remove((word, second_index))
            #Update q_find data structure with new symbol
            if new_symbol not in quick_find:
                quick_find[new_symbol] = set([(word, first_index)])
            else:
                quick_find[new_symbol].add((word, first_index))

            #Delete old info from the pairs data structure (from pairs on a boundary with the new symbol) 
            if second_index + 1 < len(segmentations[word]):
                quick_pairs[(pair[1], segmentations[word][second_index + 1])].remove((word, second_index, second_index + 1))
            if first_index - 1 >= 0: 
                quick_pairs[(segmentations[word][first_index - 1], pair[0])].remove((word, first_index - 1, first_index))
            
            #Update segmentations data structure 
            segmentations[word][first_index] = new_symbol
            segmentations[word].pop(second_index)

            #Update the pairs data structure with new pairs formed with new symbol 
            if second_index < len(segmentations[word]):
                if (new_symbol, segmentations[word][second_index]) not in quick_pairs:                quick_pairs[(new_symbol, segmentations[word][second_index])] = set([(word, first_index, second_index)])
                else:
                    quick_pairs[(new_symbol, segmentations[word][second_index])].add((word, first_index, second_index))

            if first_index - 1 >= 0:
                if (segmentations[word][first_index - 1], new_symbol) not in quick_pairs:
                    quick_pairs[(segmentations[word][first_index - 1], new_symbol)] = set([(word, first_index - 1 , first_index)])
                else:
                    quick_pairs[(segmentations[word][first_index -1], new_symbol)].add((word, first_index - 1 , first_index))

            #Now, move the indicies for things after the merged pair!
            for i in range(second_index, len(segmentations[word])):
                quick_find[segmentations[word][i]].remove((word, i + 1))
                quick_find[segmentations[word][i]].add((word, i))
                if i + 1 < len(segmentations[word]):
                    quick_pairs[(segmentations[word][i], segmentations[word][i+1])].remove((word, i + 1 , i + 2))
                    quick_pairs[(segmentations[word][i], segmentations[word][i+1])].add((word, i , i + 1))

        
        #One last thing now that we're done...
        quick_pairs.pop(pair)
    except Exception as e:
        print("Starting debug output")
        print(type(e))
        print(e.args)
        print(e)
        pdb.set_trace()


    


if __name__ == '__main__':
    #Main hyperparameter!
    gamma = 0.3

    parser = create_parser()
    args = parser.parse_args()
    word_vectors = pickle.load(args.vectors)
    segmentations = {}
    
    #Dict that maps characters to words containing them (for speed)
    quick_find = {}
    #Dict that maps pairs to words containing them (for speed)
    quick_pairs = {}
    vocab = get_vocabulary(args.input)
    #Map word and part to a score for memoization
    memoized_scores = {}

    #Each words starts totally segmented..
    #Set up to the data structures
    for word in vocab:
        #Set up segmentations data structure
        segmentations[word] = list(word)
        #Set up the quick_find data structure
        for idx, c in enumerate(word):
            if c not in quick_find:
                quick_find[c] = set([(word, idx)])
            else:
                quick_find[c].add((word, idx))
            #Now set up the quick_pairs data structure
            if idx != len(word) - 1:
                if (c, word[idx+1]) not in quick_pairs:
                    quick_pairs[(c, word[idx+1])] = set([(word, idx, idx+1)])
                else:
                    quick_pairs[(c, word[idx+1])].add((word, idx, idx+1))



    print("SIZE QUICK FIND")
    print(len(quick_find.keys()))

    print("SIZE QUICK PAIRS")
    print(len(quick_pairs.keys()))

    print("SIZE VOCAB")
    print(len(vocab.keys()))


    num_iterations = 1000
    #Core algorithm
    #Traverse pairs in random order and decide whether to merge
    seen = set()
    for i in range(num_iterations):
        print(i)
        operation = randint(0, len(quick_pairs) - 1)
        cur_pair = list(quick_pairs.keys())[operation]
        if cur_pair in seen:
            continue
        else:
            seen.add(cur_pair)
        merge_pair = get_pair_action(cur_pair)
        #DEBUG HACK!!!
        merge_pair = True
        if merge_pair:
            merge_update(cur_pair)

            


    #Write the word segmentations to the output file
    for word in vocab:
        final_seg = segmentations[word]
        delimited_seg = " ".join(final_seg)
        args.output.write(delimited_seg)
        args.output.write('\n')













        




