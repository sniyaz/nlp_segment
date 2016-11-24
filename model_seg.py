"""
Alternative to BPE based on N-gram Languge Modeling. Each character is dependent on n
before it, where n is a hyper parameter.
"""

from __future__ import unicode_literals, division

import sys
import codecs
import argparse
import string
from collections import defaultdict, Counter
from random import randint, shuffle

import numpy as np
import pickle
from itertools import combinations
from random import shuffle
import copy
import operator
from math import log

from bpe import get_vocabulary

import pdb


#Big table of all n-gram frequencies
def get_ngram_statistics(vocab, segmentations, n):
    conditional_freqs = defaultdict(lambda: Counter())
    base_freqs = Counter()
    cur_ngram = []
    
    for word in vocab:
        for char in segmentations[word]:
            cur_ngram.append(char)
            if len(cur_ngram) > n:
                ngram_tup = tuple(cur_ngram)
                conditional_freqs[ngram_tup[:-1]][ngram_tup[-1]] += vocab[word]
                base_freqs[ngram_tup[:-1]] += vocab[word]
                cur_ngram.pop(0)
                    
    return base_freqs, conditional_freqs


def get_neighbor_ngrams(word, pair_index_1, pair_index_2, segmentations, n):
    neighbor_ngrams = []
    word_seg = segmentations[word]
    new_char = word_seg[pair_index_1] + word_seg[pair_index_2]

    ngram = [new_char]
    #For when we go and calculate the ngrams that were replaced.
    temp_ngram = [word_seg[pair_index_1]]

    back_index = pair_index_1
    for i in range(n):
        back_index -= 1
        ngram.insert(0, word_seg[back_index])
        temp_ngram.insert(0, word_seg[back_index])
    
    neighbor_ngrams.append(tuple(ngram))

    #First forward pass to find all the NEW ngrams!
    front_index = pair_index_2
    
    for i in range(n):
        front_index += 1
        ngram.append(word_seg[front_index])
        ngram.pop(0)
        neighbor_ngrams.append(tuple(ngram))

    #One more "forward pass"
    #Returns the old/replaced n-grams in the changed section so new corpus prob can be calculated.
    previous_ngrams = []
    ngram = temp_ngram
    previous_ngrams.append(tuple(ngram))

    front_index = pair_index_1

    for i in range(n):
        front_index += 1
        ngram.append(word_seg[front_index])
        ngram.pop(0)
        previous_ngrams.append(tuple(ngram))
    
    #pdb.set_trace()
    return neighbor_ngrams, previous_ngrams


def get_init_merge_data(vocab, quick_pairs, segmentations, base_freqs, conditional_freqs, n):
    #future_merge_data = = defaultdict(lambda: [defaultdict(lambda: Counter()), Counter()])
    future_merge_data = Counter()
    for pair in quick_pairs:
        involved_words = quick_pairs[pair]
        changed_bases = Counter()
        changed_conditionals = defaultdict(lambda: Counter())
        for word, pair_index_1, pair_index_2 in involved_words:
            neighbor_ngrams, previous_ngrams = get_neighbor_ngrams(word, pair_index_1, pair_index_2, segmentations, n)
            #Keep a record of all that has changed.
            for ngram in previous_ngrams:
                cur_base = ngram[:-1]
                cur_conditioned = ngram[-1]
                changed_bases[cur_base] -= vocab[word]
                changed_conditionals[cur_base][cur_conditioned] -= vocab[word]
            for ngram in neighbor_ngrams:
                cur_base = ngram[:-1]
                cur_conditioned = ngram[-1]
                changed_bases[cur_base] += vocab[word]
                changed_conditionals[cur_base][cur_conditioned] += vocab[word]
                
        #Calculate the change in log probability if pair merged.
        delta = 0
        for delta_base in changed_bases:
            if changed_bases[delta_base] == 0:
                continue
            new_base_count = base_freqs[delta_base] + changed_bases[delta_base]
            for prev_conditioned in conditional_freqs[delta_base]:
                conditional_count = conditional_freqs[delta_base][prev_conditioned]
                #Undo the old probability contribution
                delta -= conditional_count*log(conditional_count/base_freqs[delta_base])
                #And add the new one....
                if new_base_count > 0:
                    delta += conditional_count*log(conditional_count/new_base_count)

        for delta_base in changed_conditionals:
            new_base_count = base_freqs[delta_base] + changed_bases[delta_base]
            if new_base_count > 0:
                cur_cond_dict = changed_conditionals[delta_base]
                for delta_conditioned in cur_cond_dict:
                    count_change = cur_cond_dict[delta_conditioned]
                    if count_change == 0:
                        continue
                    old_conditional_count = conditional_freqs[delta_base][delta_conditioned]
                    new_conditional_count = old_conditional_count + count_change
                    #Take away the old conditional probabilities...
                    if old_conditional_count > 0:
                        delta -= old_conditional_count*log(old_conditional_count/new_base_count)
                     
                    #And add the new ones!
                    if new_conditional_count > 0:
                        delta += new_conditional_count*log(new_conditional_count/new_base_count)
            
        future_merge_data[pair] = delta

    return future_merge_data


def get_segmentation_data(vocab, n):
    segmentations = {}
    #Dict that maps pairs to words containing them
    quick_pairs = defaultdict(lambda: set())

    for word in vocab:
        #Set up segmentations data structure
        seg = list(word)
        segmentations[word] = seg
        #Set up the quick_pairs data structure
        for idx, c in enumerate(seg):
            if idx != len(seg) - 1:
                quick_pairs[(c, seg[idx+1])].add((word, idx+n, idx+n+1))
        #Add padding to the edges of the word for CPT stuff
        [segmentations[word].append(None) for i in range(n)]
        [segmentations[word].insert(0, None) for i in range(n)]

    return segmentations, quick_pairs



if __name__ == '__main__':

    target_file = sys.argv[1]
    #How many characters before count in our Markov Model.
    num_before = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    
    target_object = open(target_file, "r")
    vocab = get_vocabulary(target_object)

    segmentations, quick_pairs = get_segmentation_data(vocab, num_before)
    base_freqs, conditional_freqs = get_ngram_statistics(vocab, segmentations, num_before)

    #Calculate the initial list of probability deltas for merges.
    future_merge_data = get_init_merge_data(vocab, quick_pairs, segmentations, base_freqs, conditional_freqs, num_before)
    
    pdb.set_trace()



    
   

  