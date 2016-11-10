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
import zlib
import difflib

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
        description="test and report with our algorithm")
    parser.add_argument(
        '--mode', action="store",
        help="1 -> Vanilla BPE, 2-> Tie Breaking BPE")
    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument("-ft", action="store_true",
        help="Set if input corpus is frequencey table")
    #Just to clarify, this is the START of the TWO output files' names. 
    #See the end of this file.
    parser.add_argument(
        '--output', '-o', action="store",
        metavar='PATH',
        help="Output name")
    parser.add_argument(
        '--symbols', '-s', action="store",
        help="Number of merge operations to perform")
    
    return parser


def get_vocabulary(fobj):
    fobj.seek(0)
    vocab = Counter()
    for line in fobj:
        line = line.strip()
        for word in line.split():
            vocab[word] += 1

    return vocab


def get_vocabulary_freq_table(fobj, word_vectors):
    fobj.seek(0)
    #Write out a pure corpus so we know what we trained on. "Pure" means it was in the 300k word vectors.
    # pure_corpus_obj = open(args.output + "_pure_corpus.txt", "w+")
    # exluded_corpus_obj = open(args.output + "_excluded_corpus.txt", "w+") 
    vocab = Counter()
    for line in fobj:
        original_line = line
        line = line.strip()
        line_parts = line.split()
        if len(line_parts) != 2:
            continue
        freq = int(line_parts[0])
        word = line_parts[1]
        vocab[word] += freq
       
    return vocab


def apply_presegs(vocab, presegs):
    for word in presegs:
        if len(presegs[word]) > 1:
            freq = vocab[word]
            vocab[word] -= freq
            for part in presegs[word]:
                vocab[part] += freq
    return vocab


def recover_preseg_boundary(vocab, presegs, segmentations_in):
    segmentations_out = {}
    for word in presegs:
        final_seg = []
        for part in presegs[word]:
            final_seg.extend(segmentations_in[part])
        segmentations_out[word] = final_seg
    return segmentations_out


#Needed if doing BPE with tie breaking. Big table of all pair frequencies
def get_pair_statistics(vocab, segmentations):
    all_freqs = Counter()
    for word, freq in vocab.items():
        seg = segmentations[word]
        prev_char = seg[0]
        for char in seg[1:]:
            all_freqs[(prev_char, char)] += freq
            prev_char = char

    return all_freqs


def word_similarity(word1, word2):
    vec1 = word_vectors[word1]
    vec2 = word_vectors[word2]
    return cosine_similarity(vec1, vec2)
    

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


#Euclidean distance between 2 vectors
def distance(vec1, vec2):
    difference = vec1 - vec2
    return np.linalg.norm(difference)


def get_mean(sample_set, word_vectors):
    if not sample_set:
        return 0
    
    average = None
    for sample in sample_set:
        if average is None:
            average = word_vectors[sample[0]]
        else:
            #We only care about the word in the tuple
            average = average + word_vectors[sample[0]]
           
    average = average/len(sample_set)
    return average


def get_set_cohesion(word_set, word_vectors):
    involved_words = [i[0] for i in word_set]
    pair_words = set()
    for word in involved_words:
        pair_words.add((word,))
    if len(pair_words) == 0:
        return 0
    #Center of the new symbol's vectors
    pair_mean = get_mean(pair_words, word_vectors)
    #pdb.set_trace()
    average_similarity = 0

    for word in pair_words:
        average_similarity += np.dot(word_vectors[word[0]], pair_mean)
    
    #pdb.set_trace()
    average_similarity = average_similarity/len(pair_words)
   
    return average_similarity


#Updates the quick_pairs and segmentations data structures for a given word.
def core_word_update(word, pair, new_symbol, first_index, second_index, quick_pairs, \
    segmentations, freq_changes, all_freqs, update_caches):
    #Delete old info from the pairs data structure (from pairs on a boundary with the new symbol) 
    if second_index + 1 < len(segmentations[word]):
        quick_pairs[(pair[1], segmentations[word][second_index + 1])].remove((word, second_index, second_index + 1))
        if update_caches:
            all_freqs[(pair[1], segmentations[word][second_index + 1])] -= vocab[word]
            freq_changes[(pair[1], segmentations[word][second_index + 1])] = all_freqs[(pair[1], segmentations[word][second_index + 1])]

    if first_index - 1 >= 0: 
        quick_pairs[(segmentations[word][first_index - 1], pair[0])].remove((word, first_index - 1, first_index))
        if update_caches:
            all_freqs[(segmentations[word][first_index - 1], pair[0])] -= vocab[word]
            freq_changes[(segmentations[word][first_index - 1], pair[0])] = all_freqs[(segmentations[word][first_index - 1], pair[0])]

    
    #Update segmentations data structure 
    segmentations[word][first_index] = new_symbol
    segmentations[word].pop(second_index)

    #Update the pairs data structure with new pairs formed with new symbol 
    if second_index < len(segmentations[word]):
        quick_pairs[(new_symbol, segmentations[word][second_index])].add((word, first_index, second_index))
        if update_caches:
            all_freqs[(new_symbol, segmentations[word][second_index])] += vocab[word]
            freq_changes[(new_symbol, segmentations[word][second_index])] += vocab[word]
        
    if first_index - 1 >= 0:
        quick_pairs[(segmentations[word][first_index -1], new_symbol)].add((word, first_index - 1 , first_index))
        if update_caches: 
            all_freqs[(segmentations[word][first_index - 1], new_symbol)] += vocab[word]
            freq_changes[(segmentations[word][first_index - 1], new_symbol)] += vocab[word]
    
    #Now, move the indicies for things after the merged pair!
    for i in range(second_index, len(segmentations[word]) - 1):
        quick_pairs[(segmentations[word][i], segmentations[word][i+1])].remove((word, i + 1 , i + 2))
        quick_pairs[(segmentations[word][i], segmentations[word][i+1])].add((word, i , i + 1))
        

#MASSIVE monster of a function that updates all data structures after a merge operation...
def merge_update(pair, quick_pairs, quick_find, segmentations, freq_cache, all_freqs, threshold):
    #Helper for decting when the last occurance of a character in a word vanishes
    def remove_word_check(word, in_part):
        for part in segmentations[word]:
            if part == in_part:
                return False
        return True

    new_symbol = "".join(pair)
    involved_words = quick_pairs[pair]

    #Book keeping if doing BPE tie breaking..
    freq_changes = Counter()
    #Edge cases can have you change the set as you iterate over it!
    while involved_words:
        word, first_index, second_index = involved_words.pop()

        core_word_update(word, pair, new_symbol, first_index, second_index, quick_pairs, segmentations, freq_changes, all_freqs, True)

        #Remove the mapping of the word and old symbols from the quick_find structure
        if remove_word_check(word, pair[0]):
            quick_find[pair[0]].remove((word,))
        if remove_word_check(word, pair[1]):
            #New edge case in situations like "l" + "l"
            if pair[0] != pair[1]:
                quick_find[pair[1]].remove((word,))
        #Update q_find data structure with new symbol
        quick_find[new_symbol].add((word,))
        
    #Now we have to clean up the frequencey cache...
    for changed_pair in freq_changes:
        if freq_changes[changed_pair] > threshold:
            freq_cache[changed_pair] = freq_changes[changed_pair]
        else:
            if changed_pair in freq_cache:
                freq_cache.pop(changed_pair)
    
    #Sometimes this can be an issue when the pair already got popped above.
    if pair in freq_cache:
        freq_cache.pop(pair)

    #One last thing now that we're done...
    quick_pairs.pop(pair)
    all_freqs.pop(pair)


def is_true_tie(pair_choices):
    for pair in pair_choices:
        for other_pair in pair_choices:
            if other_pair != pair and (pair[0] == other_pair[1] or pair[1] ==  other_pair[0]):
                return True
    return False


def deterministic_hash(data):
    data = str(data).encode('utf-8')
    return zlib.adler32(data)


def draw_random_pairs():
    pairs = list(quick_pairs.keys())
    drawn_indicies = [randint(0, len(pairs) - 1) for i in range(search_scatter)]
    drawn_pairs = [pairs[i] for i in drawn_indicies]
    return drawn_pairs


#Check the freq_cache and refresh if needed.
def check_cache(freq_cache, threshold, all_freqs, iter_num):
    if len(freq_cache) == 0:
        most_frequent = max(all_freqs, key=all_freqs.get)
        new_threshold = all_freqs[most_frequent] * iter_num/(iter_num+10000.0)
        refresh_freq_cache(freq_cache, new_threshold, all_freqs)
        return new_threshold
    else:
        return threshold    
    
    
def refresh_freq_cache(freq_cache, new_threshold, all_freqs):
    freq_cache.clear()
    for pair in all_freqs:
        if all_freqs[pair] > new_threshold:
            freq_cache[pair] = all_freqs[pair]


#Inspired by the BPE Paper code. Remember to cite if needed.
def draw_frequent_pairs(freq_cache):
    frequent_pair = max(freq_cache, key=freq_cache.get)
    most_frequent_pairs = [p for p in freq_cache if freq_cache[p] == freq_cache[frequent_pair]]
    shuffle(most_frequent_pairs)
    return most_frequent_pairs


def segment_vocab(vocab, num_iterations):
    segmentations = {}
    
    #Dict that maps characters to words containing them
    quick_find = defaultdict(lambda: set())
    #Dict that maps pairs to words containing them
    quick_pairs = defaultdict(lambda: set())
    #Cache for speeing up max when finding frequent pairs
    freq_cache = Counter()

    #Each words starts totally segmented..
    #Set up to the data structures
    for word in vocab:
        #Set up segmentations data structure
        seg = list(word)
        #seg.append("</w>")
        segmentations[word] = seg
        #Set up the quick_find data structure
        for idx, c in enumerate(seg):
            quick_find[c].add((word,))

            #Now set up the quick_pairs data structure
            if idx != len(seg) - 1:
                quick_pairs[(c, seg[idx+1])].add((word, idx, idx+1))

    all_freqs = get_pair_statistics(vocab, segmentations)
    #Set initial threshold and populate freq_cache
    threshold = all_freqs[max(all_freqs, key=all_freqs.get)]/10
    refresh_freq_cache(freq_cache, threshold, all_freqs)
    
    num_ties = 0
    merges_done = []

    #Core algorithm
    for i in range(num_iterations):
        
        #Has the max frequencey cache gone dry?
        threshold = check_cache(freq_cache, threshold, all_freqs, i)

        drawn_pairs = draw_frequent_pairs(freq_cache)            
        best_pair = drawn_pairs[0]

        sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, best_pair[0], best_pair[1], freq_cache[best_pair]))                           
        
        merge_update(best_pair, quick_pairs, quick_find, segmentations, freq_cache, all_freqs, threshold)
        merges_done.append(best_pair)

    return segmentations, merges_done



if __name__ == '__main__':
         
    parser = create_parser()
    args = parser.parse_args()

    word_vectors = pickle.load(open("/Users/Sherdil/Research/NLP/nlp_segment/data/vectors.txt", "rb"))
    
    #Dict that maps characters to words containing them
    quick_find = defaultdict(lambda: set())
    #Dict that maps pairs to words containing them
    quick_pairs = defaultdict(lambda: set())
    #Cache for speeing up max when finding frequent pairs
    threshold = None
    freq_cache = Counter()
    if args.ft:
        vocab = get_vocabulary_freq_table(args.input, word_vectors)
    else:
        vocab = get_vocabulary(args.input)
    
    mode = int(args.mode)
    #Vanilla
    if mode == 1: 
        pass
    #Morpology Pre-Segmentation
    elif mode == 2:
        with open("../debug_temp/presegs_ckpt.txt", "rb") as checkpoint_file:
            presegs = pickle.load(checkpoint_file)
        vocab = apply_presegs(vocab, presegs)
        for word in list(vocab.keys()):
            if vocab[word] == 0:
                vocab.pop(word)

    num_iterations = int(args.symbols)
    #Invoke Main BPE Algo
    segmentations, merges_done = segment_vocab(vocab, num_iterations)

    #Write out the segmentations of each word in the corpus.
    segs_output_obj = open(args.output + "_segs.txt", "w+")
    to_write = list(vocab.keys())
    to_write.sort()
    #Write the word segmentations to the output file
    for word in to_write:
        final_seg = segmentations[word]
        delimited_seg = " ".join(final_seg)
        segs_output_obj.write(word + ": " + delimited_seg)
        segs_output_obj.write('\n')
    segs_output_obj.close()


    #Write out the merge operations in the order they were done.
    merge_ops_output_obj = open(args.output + "_merge_ops.txt", "w+")
    for pair in merges_done:
        merge_ops_output_obj.write(" ".join(pair))
        merge_ops_output_obj.write('\n')
    merge_ops_output_obj.close()













        




