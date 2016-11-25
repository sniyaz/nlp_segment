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


def get_neighbor_ngrams(word_seg, pair_index_1, pair_index_2, n):
    neighbor_ngrams = []
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


#Return delta information for the pairs in the given pairs_list.
def get_pair_deltas(pairs_list, vocab, quick_pairs, segmentations, base_freqs, conditional_freqs, n):
    merge_deltas = Counter()
    merge_changed_bases = {}
    merge_changed_conditionals = {}
    for pair in pairs_list:
        involved_words = quick_pairs[pair]
        changed_bases = Counter()
        changed_conditionals = defaultdict(lambda: Counter())
        for word, pair_index_1, pair_index_2 in involved_words:
            neighbor_ngrams, previous_ngrams = get_neighbor_ngrams(segmentations[word], pair_index_1, pair_index_2, n)
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
        delta = calculate_delta(changed_bases, changed_conditionals, base_freqs, conditional_freqs)     
        merge_deltas[pair] = delta
        merge_changed_bases[pair] = changed_bases
        merge_changed_conditionals[pair] = changed_conditionals

    return merge_deltas, merge_changed_bases, merge_changed_conditionals


def calculate_delta(changed_bases, changed_conditionals, base_freqs, conditional_freqs):
    #Calculate the change in log probability if pair merged.
    delta = 0
    for delta_base in changed_bases:
        if changed_bases[delta_base] == 0:
            continue
        old_base_count = base_freqs[delta_base]
        new_base_count =  old_base_count + changed_bases[delta_base]
        if old_base_count > 0:
            delta += old_base_count*log(old_base_count)
        if new_base_count > 0:
            delta -= old_base_count*log(new_base_count)
        

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
    return delta


#Modified from bpe.py to account for None characters.
def core_word_update(word, pair, first_index, second_index, quick_pairs, segmentations, n):
    new_symbol = "".join(pair)
    #Delete old info from the pairs data structure (from pairs on a boundary with the new symbol) 
    if second_index + 1 <= len(segmentations[word]) - n - 1:
        quick_pairs[(pair[1], segmentations[word][second_index + 1])].remove((word, second_index, second_index + 1))
       
    if first_index - 1 >= n: 
        quick_pairs[(segmentations[word][first_index - 1], pair[0])].remove((word, first_index - 1, first_index))
        
    #Update segmentations data structure 
    segmentations[word][first_index] = new_symbol
    segmentations[word].pop(second_index)

    #Update the pairs data structure with new pairs formed with new symbol 
    if second_index <= len(segmentations[word]) - n - 1:
        quick_pairs[(new_symbol, segmentations[word][second_index])].add((word, first_index, second_index))
        
    if first_index - 1 >= n:
        quick_pairs[(segmentations[word][first_index -1], new_symbol)].add((word, first_index - 1 , first_index))
    
    #Now, move the indicies for things after the merged pair!
    for i in range(second_index, len(segmentations[word]) - n - 1):
        quick_pairs[(segmentations[word][i], segmentations[word][i+1])].remove((word, i + 1 , i + 2))
        quick_pairs[(segmentations[word][i], segmentations[word][i+1])].add((word, i , i + 1))
        

def apply_merge(pair, vocab, quick_pairs, segmentations, merge_deltas, merge_changed_bases, merge_changed_conditionals, base_freqs, conditional_freqs, n):

    new_changed_bases = merge_changed_bases[pair]
    new_changed_conditionals = merge_changed_conditionals[pair]

    old_base_freqs = copy.deepcopy(base_freqs)
    old_conditional_freqs = copy.deepcopy(conditional_freqs)

    #Which pairs need to be recalculated later?
    invalidated_pairs = get_invalidated_pairs(pair, quick_pairs,segmentations, n)
    
    #Update our frequency data_structures
    for delta_base in new_changed_bases:
        base_freqs[delta_base] += new_changed_bases[delta_base]
        #Prune bases that don't exist anymore.
        if base_freqs[delta_base] == 0:
            base_freqs.pop(delta_base)
            conditional_freqs.pop(delta_base)
        
    for delta_base in new_changed_conditionals:
        cur_cond_dict = new_changed_conditionals[delta_base]
        for delta_conditioned in cur_cond_dict:
            conditional_freqs[delta_base][delta_conditioned] += cur_cond_dict[delta_conditioned]
    
    #Update segmentation data_structures
    involved_words = quick_pairs[pair]
    while involved_words:
        word, first_index, second_index = involved_words.pop()
        core_word_update(word, pair, first_index, second_index, quick_pairs, segmentations, n)

    #Done with this pair: remove from data_structures.
    invalidated_pairs.remove(pair)
    quick_pairs.pop(pair)
    merge_deltas.pop(pair)
    merge_changed_bases.pop(pair)
    merge_changed_conditionals.pop(pair)
    
    print("START")
    #RESET the old contribution to the delta.
    for pair in merge_deltas:
        pair_base_changes = merge_changed_bases[pair]
        pair_conditional_changes = merge_changed_conditionals[pair]

        for delta_base in pair_base_changes:
            if delta_base not in pair_base_changes:
                continue
            old_base_count = old_base_freqs[delta_base]
            expected_base_count =  old_base_count + pair_base_changes[delta_base]
            if old_base_count > 0:
                merge_deltas[pair] -= old_base_count*log(old_base_count)
            if expected_base_count > 0:
                merge_deltas[pair] += old_base_count*log(expected_base_count)
            
        for delta_base in new_changed_conditionals:
            cur_cond_dict = new_changed_conditionals[delta_base]
            expected_base_count = old_base_freqs[delta_base] + pair_base_changes[delta_base]
            if expected_base_count > 0:
                for delta_conditioned in cur_cond_dict:
                    if delta_conditioned in pair_conditional_changes[delta_base]:     
                        count_change = pair_conditional_changes[delta_base][delta_conditioned]
                        if count_change == 0:
                            continue
                        old_conditional_count = old_conditional_freqs[delta_base][delta_conditioned]
                        expected_conditional_count = old_conditional_count + count_change
                        if old_conditional_count > 0:
                            merge_deltas[pair] += old_conditional_count*log(old_conditional_count/expected_base_count)
                        if expected_conditional_count > 0:
                            merge_deltas[pair] -= expected_conditional_count*log(expected_conditional_count/expected_base_count)
    print("END")

    pdb.set_trace()

    #Finally, REDO the delta calculations for the pairs that had their entries invalidated.
    updated_deltas, updated_changed_bases, updated_changed_conditionals = get_pair_deltas(invalidated_pairs, vocab, quick_pairs, segmentations, base_freqs, conditional_freqs, n)
    for pair in updated_deltas:
        merge_deltas[pair] = updated_deltas[pair]
        merge_changed_bases[pair] = updated_changed_bases[pair]
        merge_changed_conditionals[pair] = updated_changed_conditionals[pair]


def get_invalidated_pairs(pair, quick_pairs, segmentations, n):
    invalidated_pairs = set()
    involved_words = quick_pairs[pair]
    for word, first_index, second_index in involved_words:
        word_seg = segmentations[word]
        #Get the newly created pairs that also need delta values.
        if first_index - 1 >= n:
            invalidated_pairs.add((word_seg[first_index - 1], "".join(pair)))
        if second_index + 1 <= len(word_seg) - 1 - n:
            invalidated_pairs.add(("".join(pair), word_seg[second_index + 1]))
        #Also, all the old pairs that were disrupted.
        back_index = max(first_index - n, n)
        front_index = back_index + 1
        cut_off = min(len(word_seg) - 1 - n, second_index + n)
        while front_index <= cut_off:
            cur_pair = (word_seg[back_index], word_seg[front_index])
            invalidated_pairs.add(cur_pair)
            back_index += 1
            front_index += 1

    return invalidated_pairs


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

    #Calculate the initial set of deltas for merges.
    merge_deltas, merge_changed_bases, merge_changed_conditionals = get_pair_deltas(quick_pairs, vocab, quick_pairs, segmentations, base_freqs, conditional_freqs, num_before)
    
    for i in range(num_iterations):
        best_pair = max(merge_deltas, key=lambda x: merge_deltas[x])
        sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (log_prob_delta {3})\n'.format(i, best_pair[0], best_pair[1], merge_deltas[best_pair]))
        apply_merge(best_pair, vocab, quick_pairs, segmentations, merge_deltas, merge_changed_bases, merge_changed_conditionals, base_freqs, conditional_freqs, num_before)
        



    
   

  