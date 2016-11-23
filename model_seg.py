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
def get_ngram_statistics(target_object, n):
    target_object.seek(0)
    ngram_freqs = defaultdict(lambda: Counter())
    conditional_freqs = Counter()
    cur_ngram = []

    for line in target_object:
        line = line.strip()
        for word in line.split():
            for char in word:
                cur_ngram.append(char)
                if len(cur_ngram) > n:
                    ngram_tup = tuple(cur_ngram)
                    ngram_freqs[ngram_tup[:-1]][ngram_tup[-1]] += 1
                    conditional_freqs[ngram_tup[:-1]] += 1
                    cur_ngram.pop(0)
                    
    return ngram_freqs, conditional_freqs


#Represents corpus as continuous list and also gets dict mapping words to indices in that list.
#Needed now since we need to get information from previous words as well, unlike BPE.
def get_word_indices(target_object):
    target_object.seek(0)
    corpus_list = []
    quick_find_words = defaultdict(lambda: set())
    quick_find_pairs = defaultdict(lambda: set())
    
    i = 0
    for line in target_object:
        line = line.strip()
        for word in line.split():
            corpus_list.append(word)
            quick_find_words[word].add(i)
            prev_char = word[0]       
            for idx, c in enumerate(word):
                #Now set up the quick_find_pairs data structure
                if idx != len(word) - 1:
                    quick_find_pairs[(c, word[idx+1])].add((word, idx, idx+1))

            i += 1
    
    return corpus_list, quick_find_words, quick_find_pairs


#Calculate the initial log probability of the corpus.
def get_init_prob(target_object, ngram_freqs, conditional_freqs, n):
    target_object.seek(0)
    #NOTE: This is a log prob to prevent underflow.
    init_prob = 0
    cur_ngram = []

    for line in target_object:
        line = line.strip()
        for word in line.split():
            for char in word:
                cur_ngram.append(char)
                if len(cur_ngram) > n:
                    ngram_tup = tuple(cur_ngram)
                    base_count = conditional_freqs[ngram_tup[:-1]]
                    ngram_count = ngram_freqs[ngram_tup[:-1]][ngram_tup[-1]]
                    conditional_prob = float(ngram_count)/float(base_count)
                    init_prob += log(conditional_prob)
                    cur_ngram.pop(0)
                    
    return init_prob


def get_neighbor_ngrams(word, pair_index_1, pair_index_2, n, word_index, corpus_list):
    neighbor_ngrams = []
    new_char = word[pair_index_1] + word[pair_index_2]

    back_word = word
    back_word_index = word_index

    ngram = [new_char]
    #For when we go and calculate the ngrams that were replaced.
    temp_ngram = [word[pair_index_1]]

    back_index = pair_index_1
    for i in range(n):
        back_index -= 1
        if back_index < 0:
            back_word_index -=1
            back_word = corpus_list[back_word_index]
            back_index = len(back_word) - 1
        ngram.insert(0, back_word[back_index])
        temp_ngram.insert(0, back_word[back_index])
    
    neighbor_ngrams.append(tuple(ngram))

    #First forward pass to find all the NEW ngrams!
    front_index = pair_index_2
    front_word = word
    front_word_index = word_index
    
    for i in range(n):
        front_index += 1
        if front_index >= len(front_word):
            front_word_index += 1
            front_word = corpus_list[front_word_index]
            front_index = 0
        ngram.append(front_word[front_index])
        ngram.pop(0)
        neighbor_ngrams.append(tuple(ngram))

    #One more "forward pass"
    #Returns the old/replaced n-grams in the changed section so new corpus prob can be calculated.
    previous_ngrams = []
    ngram = temp_ngram
    previous_ngrams.append(tuple(ngram))

    front_index = pair_index_1
    front_word = word
    front_word_index = word_index

    for i in range(n+1):
        front_index += 1
        if front_index >= len(front_word):
            front_word_index += 1
            front_word = corpus_list[front_word_index]
            front_index = 0
        ngram.append(front_word[front_index])
        ngram.pop(0)
        previous_ngrams.append(tuple(ngram))
    
    #pdb.set_trace()
    return neighbor_ngrams, previous_ngrams


def get_init_merge_data(corpus_list, quick_find_words, quick_find_pairs, ngram_freqs, conditional_freqs, n):
    #future_merge_data = = defaultdict(lambda: [defaultdict(lambda: Counter()), Counter()])
    future_merge_data = Counter()
    for pair in quick_find_pairs:
        involved_words = quick_find_pairs[pair]
        changed_bases = Counter()
        changed_conditionals = defaultdict(lambda: Counter())
        for word, pair_index_1, pair_index_2 in involved_words:
            involved_indices = quick_find_words[word]
            for word_index in involved_indices:
                neighbor_ngrams, previous_ngrams = get_neighbor_ngrams(word, pair_index_1, pair_index_2, n, word_index, corpus_list)
                #Keep a record of all that has changed.
                for ngram in previous_ngrams:
                    cur_base = ngram[:-1]
                    cur_conditioed = ngram[-1]
                    changed_bases[cur_base] -= 1
                    changed_conditionals[cur_base][cur_conditioed] -= 1
                for ngram in neighbor_ngrams:
                    cur_base = ngram[:-1]
                    cur_conditioed = ngram[-1]
                    changed_bases[cur_base] += 1
                    changed_conditionals[cur_base][cur_conditioed] += 1
        
        #Calculate the change in log probability if pair merged.
        delta = 0
        for delta_base in changed_bases:
            if changed_bases[delta_base] == 0:
                continue
            new_base_count = ngram_freqs[delta_base] + changed_bases[delta_base]
            for prev_conditioned in conditional_freqs[delta_base]:
                conditional_count = conditional_freqs[delta_base][prev_conditioned]
                #Undo the old probability contribution
                delta -= conditional_count*log(conditional_count/ngram_freqs[delta_base])
                #And add the new one....
                delta += conditional_count*log(conditional_count/new_base_count)

        for delta_base in changed_conditionals:
            new_base_count = ngram_freqs[delta_base] + changed_bases[delta_base]
            cur_cond_dict = changed_conditionals[delta_base]
            for delta_conditioned in cur_cond_dict:
                count_change = cur_cond_dict[delta_conditioned]
                if count_change == 0:
                    continue
                old_conditional_count = conditional_freqs[delta_base][delta_conditioned]
                new_conditional_count = old_conditional_count + count_change
                #Take away the old conditional probabilities...
                delta -= old_conditional_count*log(old_conditional_count/new_base_count)
                #And add the new ones!
                delta += new_conditional_count*log(new_conditional_count/new_base_count)
            
        future_merge_data[pair] = delta


if __name__ == '__main__':

    target_file = sys.argv[1]
    #How many characters before count in our Markov Model.
    num_before = int(sys.argv[2])
    num_iterations = int(sys.argv[3])
    
    #TODO: Decide if needed!
    #Dict that maps characters to words containing them
    quick_find = defaultdict(lambda: set())
    #Cache for speeing up max when finding frequent pairs
    threshold = None
    freq_cache = Counter()
    
    target_object = open(target_file, "r")
    vocab = get_vocabulary(target_object)

    ngram_freqs, conditional_freqs = get_ngram_statistics(target_object, num_before)
    corpus_list, quick_find_words, quick_find_pairs = get_word_indices(target_object)

    #Calculate the initial log probability of the corpus.
    #TODO: Uncomment
    #init_prob = get_init_prob(target_object, ngram_freqs, conditional_freqs, num_before)

    #Calculate the initial list of probability deltas for merges.
    future_merge_data = get_init_merge_data(corpus_list, quick_find_words, quick_find_pairs, ngram_freqs, conditional_freqs, num_before)
    
    pdb.set_trace()



    
   

  