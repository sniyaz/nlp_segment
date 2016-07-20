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
import operator

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


def get_vocabulary(fobj):
    """Set up our data structures...
    """
    fobj.seek(0)
    vocab = Counter()
    for line in fobj:
        for word in line.split():
            vocab[word] += 1

    return vocab


def get_similarity(word1, word2):
    return np.dot(word_vectors[word1], word_vectors[word2])


def get_pair_freq(pair):
        return len(quick_pairs[pair])


#Euclidean distance between 2 vectors
def distance(vec1, vec2):
    difference = vec1 - vec2
    return np.linalg.norm(difference)


#Is merging a pair worth it? Use a sampling approach to decide...
def sample_pair_delta(pair):
    involved_words = [i[0] for i in quick_pairs[pair]]
    #We only SAMPLE the change in cost for runtime reasons
    pair_words = set()
    for word in involved_words:
        pair_words.add((word,))
    #Estimated center of the new symbol's vectors
    pair_mean = sample_mean(pair_words)

    cost_delta = 0
    for word in involved_words:
        #On a merge, the length goes down by one. Reflect the first term of the cost function
        local_delta = -1
        #"Undo" the contributions to the second part of the cost function
        cur_word_vec = word_vectors[word]
        local_delta -= gamma*distance(cur_word_vec, sample_mean(quick_find[pair[0]]))
        local_delta -= gamma*distance(cur_word_vec, sample_mean(quick_find[pair[1]]))
        #But there are new distance terms.
        local_delta += gamma*distance(cur_word_vec, pair_mean)

        cost_delta += vocab[word]*local_delta
            
    return cost_delta


def sample_mean(sample_set):
    sampled_words = sample_words(sample_set)
    if not sampled_words:
        return 0
    
    average = None
    for sample in sampled_words:
        if  average is None:
            average = word_vectors[sample[0]]
        else:
            #We only care about the word in the tuple
            average = average + word_vectors[sample[0]]
           
    average = average/len(sampled_words)
    return average


def sample_words(sample_set):
    sampled_words = []
    for i in range(sample_size):
        if not sample_set:
            break
        sample = sample_set.pop()
        sampled_words.append(sample)
    
    #We also want to RETURN the items to the set..
    for sample in sampled_words:
        sample_set.add(sample)
        
    return sampled_words


#Needed if doing BPE with tie breaking. Big table of all pair frequencies
def get_pair_statistics(vocab):
    for i, (word, freq) in enumerate(vocab):
        prev_char = word[0]
        for char in word[1:]:
            all_freqs[prev_char, char] += freq

    return stats, indices


#MASSIVE monster of a function that updates all data structures after a merge operation...
def merge_update(pair):

    #Helper for decting when the last occurance of a character in a word vanishes
    def remove_word_check(word, in_part):
        for part in segmentations[word]:
            if part == in_part:
                return False
        return True

    new_symbol = "".join(pair)
    involved_words = quick_pairs[pair]

    #Edge cases can have you change the set as you iterate over it!
    while involved_words:
        word, first_index, second_index = involved_words.pop()

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
            if (new_symbol, segmentations[word][second_index]) not in quick_pairs:                
                quick_pairs[(new_symbol, segmentations[word][second_index])] = set([(word, first_index, second_index)])
            else:
                quick_pairs[(new_symbol, segmentations[word][second_index])].add((word, first_index, second_index))

        if first_index - 1 >= 0:
            if (segmentations[word][first_index - 1], new_symbol) not in quick_pairs:
                quick_pairs[(segmentations[word][first_index - 1], new_symbol)] = set([(word, first_index - 1 , first_index)])
            else:
                quick_pairs[(segmentations[word][first_index -1], new_symbol)].add((word, first_index - 1 , first_index))
        
        #Now, move the indicies for things after the merged pair!
        for i in range(second_index, len(segmentations[word]) - 1):
            quick_pairs[(segmentations[word][i], segmentations[word][i+1])].remove((word, i + 1 , i + 2))
            quick_pairs[(segmentations[word][i], segmentations[word][i+1])].add((word, i , i + 1))
        
        #Remove the mapping of the word and old symbols from the quick_find structure
        if remove_word_check(word, pair[0]):
            quick_find[pair[0]].remove((word,))
        if remove_word_check(word, pair[1]):
            #New edge case in situations like "l" + "l"
            if pair[0] != pair[1]:
                quick_find[pair[1]].remove((word,))
        #Update q_find data structure with new symbol
        if new_symbol not in quick_find:
            quick_find[new_symbol] = set([(word,)])
        else:
            quick_find[new_symbol].add((word,))
        
    #One last thing now that we're done...
    quick_pairs.pop(pair)


def draw_random_pairs():
    pairs = list(quick_pairs.keys())
    drawn_indicies = [randint(0, len(pairs) - 1) for i in range(search_scatter)]
    drawn_pairs = [pairs[i] for i in drawn_indicies]
    return drawn_pairs


#Inspired by the BPE Paper code. Remember to cite if needed.
def draw_frequent_pairs():
    #Repopulate the cache in the event that it goes dry
    if len(freq_cache) == 0:
        all_pairs = list(quick_pairs.keys())
        most_frequent = max(all_pairs, key=get_pair_freq)
        threshold = get_pair_freq(most_frequent)/10
        for pair in all_pairs:
            if get_pair_freq(pair) > threshold:
                freq_cache[pair] = get_pair_freq(pair)
        
    frequent_pair, highest_freq = max(freq_cache, key=freq_cache.get)
    most_frequent_pairs = [p for p in freq_cache if freq_cache[p] == highest_freq]
    return most_frequent_pairs

        
    
if __name__ == '__main__':
    tie_break_only = False
    #Main hyperparameters!
    sample_size = 100
    #Only if employing as more than just tie-breaker
    gamma = 0.3
    search_scatter = 100

    
    parser = create_parser()
    args = parser.parse_args()
    word_vectors = pickle.load(args.vectors)
    segmentations = {}
    
    #Dict that maps characters to words containing them
    quick_find = {}
    #Dict that maps pairs to words containing them
    quick_pairs = {}
    freq_cache = Counter()
    all_freqs = Counter()
    vocab = get_vocabulary(args.input)

    #Each words starts totally segmented..
    #Set up to the data structures
    for word in vocab:
        #Set up segmentations data structure
        segmentations[word] = list(word)
        #Set up the quick_find data structure
        for idx, c in enumerate(word):
            if c not in quick_find:
                quick_find[c] = set([(word,)])
            else:
                quick_find[c].add((word,))

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

    num_iterations = 10000
    #Core algorithm
    for i in range(num_iterations):
        print(i)
        #Look at many merges, and then pick the best one
        drawn_pairs = draw_random_pairs()
        best_pair  = min(drawn_pairs, key=sample_pair_delta)        
        merge_update(best_pair)

    
    to_write = list(vocab.keys())
    to_write.sort()
    #Write the word segmentations to the output file
    for word in to_write:
        final_seg = segmentations[word]
        delimited_seg = "  ".join(final_seg)
        args.output.write(word + ": " + delimited_seg)
        args.output.write('\n')













        




