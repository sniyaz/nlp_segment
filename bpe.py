"""
Our own BPE implementation. Breaks process up into steps that make running experiments more easy. Also supports
use of pre-segmentations.
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
        help="1 -> Vanilla BPE, 2-> BPE with pre-segmentations")
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
    """
    Returns the vocab dictionary (map of word -> freq) from an object file.

    Arguments:
    fobj -- corpus file object
    """
    fobj.seek(0)
    vocab = Counter()
    for line in fobj:
        line = line.strip()
        for word in line.split():
            vocab[word] += 1

    return vocab


def get_vocabulary_freq_table(fobj, word_vectors):
    """
    Returns the vocab dictionary (map of word -> freq) from an object file. However, 
    the file is just a list of (word, freq) pairs instead of a full passage.

    Arguments:
    fobj -- corpus file object
    """
    fobj.seek(0)
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
    """
    Applies pre-segmentations to a given vocab. Non-destructive method.

    Arguments:
    vocab -- Dict of (word -> freq)
    presegs -- Dict of word -> pre-segmentations (A pre-segmentation is 
    represented as a list of word parts.)

    Returns:
    New vocab where each word has been split up by pre-segmentation.
    """
    vocab = copy.deepcopy(vocab)   
    for word in presegs:
        if word in vocab and len(presegs[word]) > 1:
            freq = vocab[word]
            vocab[word] -= freq
            for part in presegs[word]:
                vocab[part] += freq
    return vocab


def recover_preseg_boundary(vocab, presegs, segmentations_in):
    """
    Returns final set of segmentations, restoring words that were split up and 
    eliminated from the vocabulary due to pre-segmentation.

    Arguments:
    vocab -- Dict of (word -> freq). Should be a vocab processed by apply_presegs.
    presegs -- Dict of word -> pre-segmentations (A pre-segmentation is 
    represented as a list of word parts.)
    segmentations_in -- segmentations for the input vocab. Dict of word -> list of word parts.

    Returns:
    Final segmentations for words in the ORIGINAL vocab (not the input one), which obey the 
    boundaries from pre-segmentations. Note that this is a new object.

    Note:
    If you notice, our hacky way of enforcing these "pre-seg boundaries" is just splitting each word in the vocab into
    pre-seg components, and then re-combining at the en. 
    """
    segmentations_out = {}
    for word in vocab:
        if word in presegs:
            final_seg = []
            for part in presegs[word]:
                final_seg.extend(segmentations_in[part])
            segmentations_out[word] = final_seg
        else:
            segmentations_out[word] = segmentations_in[word]
    return segmentations_out


def write_segmentation_list(out_name, vocab, segmentations):
    """
    Writes a text file that contains a list of words in the vocab as well 
    as their segmentations. Words in the output file are in alphabetical order.

    Arguments:
    out_name -- file name for the output file
    vocab -- Dict of (word -> freq).
    segmentations -- segmentations for the input vocab. Dict of word -> list of word parts.

    Returns:
    None
    """
    segmentations = remove_eols(segmentations)
    #Write out the segmentations of each word in the corpus.
    segs_output_obj = open(out_name + "_segs.txt", "w+")
    to_write = list(vocab.keys())
    to_write.sort()
    #Write the word segmentations to the output file
    for word in to_write:
        delimited_seg = " ".join(segmentations[word])
        segs_output_obj.write(word + ": " + delimited_seg)
        segs_output_obj.write('\n')
    segs_output_obj.close()


def remove_eols(segmentations):
    """
    Strip all eol markers from the segmentations. Uses when BPE algo is run with eol markers.
    This method is destructive.

    Arguments:
    segmentations -- Dict of word -> list of word parts.

    Returns:
    segmentations -- same as input, but modified.
    """
    for word in segmentations.keys():
        final_seg = segmentations[word]
        if final_seg[-1] == '</w>':
            final_seg = final_seg[:-1]
        elif final_seg[-1].endswith('</w>'):
            final_seg = final_seg[:-1] + [final_seg[-1].replace('</w>','')]
        segmentations[word] = final_seg
    return segmentations


def get_pair_statistics(vocab, segmentations):
    """
    Strip all eol markers from the segmentations. Uses when BPE algo is run with eol markers.
    This method is destructive.

    Arguments:
    segmentations -- Dict of word -> list of word parts.

    Returns:
    segmentations -- same as input, but modified.
    """
    all_freqs = Counter()
    for word, freq in vocab.items():
        seg = segmentations[word]
        prev_char = seg[0]
        for char in seg[1:]:
            all_freqs[(prev_char, char)] += freq
            prev_char = char

    return all_freqs
    

def core_word_update(vocab, word, pair, new_symbol, first_index, second_index, quick_pairs, \
    segmentations, freq_changes, all_freqs, update_caches):
    """
    Updates the quick_pairs and segmentations data structures for a given word once a new merge has been 
    decided on. This is a sub-routine called by merge_update, in order to make the logic neater.

    Arguments:
    vocab -- Dict of (word -> freq).
    word -- String: word that update is focused on. 
    pair -- tuple of symbols (bigram) that was merged.
    new_symbol -- new symbol that was formed by the merge.
    first_index -- index of the first symbol in word.
    second_index -- index of the second symbol in word.
    quick_pairs -- Dict of bigram (as tuple) -> list of words containing that bigram.
    segmentations -- Dict of word -> list of word parts.
    freq_changes -- Dict that maps pair -> delta in frequency. Passed in and modified for the caller
    of this function.
    all_freqs -- Dict that maps pair -> frequency, for all pairs that currently exist. Passed in and modified
    for the caller of this function.
    update_caches -- Boolean that tells use whether to modify freq_changes and all_freqs.

    Returns:
    None. Function only modifies input data structures.
    """
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
def merge_update(vocab, pair, quick_pairs, quick_find, segmentations, freq_cache, all_freqs, threshold):
    """
    Updates the quick_pairs and segmentations data structures for a given word once a new merge has been 
    decided on. This is a sub-routine called by merge_update, in order to make the logic neater.

    Arguments:
    vocab -- Dict of (word -> freq).
    word -- String: word that update is focused on. 
    pair -- tuple of symbols (bigram) that was merged.
    new_symbol -- new symbol that was formed by the merge.
    first_index -- index of the first symbol in word.
    second_index -- index of the second symbol in word.
    quick_pairs -- Dict of bigram (as tuple) -> list of words containing that bigram.
    segmentations -- Dict of word -> list of word parts.
    freq_changes -- Dict that maps pair -> delta in frequency. Passed in and modified for the caller
    of this function.
    all_freqs -- Dict that maps pair -> frequency, for all pairs that currently exist. Passed in and modified
    for the caller of this function.
    update_caches -- Boolean that tells use whether to modify freq_changes and all_freqs.

    Returns:
    None. Function only modifies input data structures.
    """
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

        core_word_update(vocab, word, pair, new_symbol, first_index, second_index, quick_pairs, segmentations, freq_changes, all_freqs, True)

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


def check_cache(freq_cache, threshold, all_freqs, iter_num):
    """
    Check the freq_cache and refresh it if needed.

    Arguments:
    freq_cache -- Dict of bigram (as tuple) -> frequency, but only for the most frequent bigrams 
    (speeds up max operation to find most frequent bigram). Potentially modified by this function.
    threshold -- old frequency threshold that bigrams had to pass to be in the freq_cache.
    all_freqs -- Dict of bigram (as tuple) -> frequency, but for all bigrams that exist.
    iter_num -- number of merge operations that we've done.

    Returns:
    Returns: the new threshold that bigrams must pass to be in the freq_cache.
    """
    if len(freq_cache) == 0:
        most_frequent = max(all_freqs, key=all_freqs.get)
        new_threshold = all_freqs[most_frequent] * iter_num/(iter_num+10000.0)
        refresh_freq_cache(freq_cache, new_threshold, all_freqs)
        return new_threshold
    else:
        return threshold    
    
    
def refresh_freq_cache(freq_cache, new_threshold, all_freqs):
    """
    Refresh the freq_cache by searching all_freqs for pairs that are more frequent 
    than new_threshold.

    Arguments:
    freq_cache -- Dict of bigram (as tuple) -> frequency, but only for the most frequent bigrams 
    (speeds up max operation to find most frequent bigram). Potentially modified by this function.
    threshold -- old frequency threshold that bigrams had to pass to be in the freq_cache.
    all_freqs -- Dict of bigram (as tuple) -> frequency, but for all bigrams that exist.

    Returns:
    None. freq_cache is modified in-place.
    """
    freq_cache.clear()
    for pair in all_freqs:
        if all_freqs[pair] > new_threshold:
            freq_cache[pair] = all_freqs[pair]


def draw_frequent_pairs(freq_cache):
    """
    Refresh the freq_cache by searching all_freqs for pairs that are more frequent 
    than new_threshold. Pairs are shuffled in a random order before being returned.

    Arguments:
    freq_cache -- Dict of bigram (as tuple) -> frequency, but only for the most frequent bigrams 
    (speeds up max operation to find most frequent bigram). Potentially modified by this function.
    threshold -- old frequency threshold that bigrams had to pass to be in the freq_cache.
    all_freqs -- Dict of bigram (as tuple) -> frequency, but for all bigrams that exist.

    Returns:
    None. freq_cahce is modified in-place.
    """
    frequent_pair = max(freq_cache, key=freq_cache.get)
    most_frequent_pairs = [p for p in freq_cache if freq_cache[p] == freq_cache[frequent_pair]]
    shuffle(most_frequent_pairs)
    return most_frequent_pairs


def apply_merge_ops(vocab, merge_operations, num_symbols=None, use_eol=False):
    """
    Take a list of trained merge operations a apply them to a new vocabulary! This isn't part of the 
    standard BPE pipeline, but can be called by other scripts that to force certain merges.

    Arguments:
    vocab -- Dict of (word -> freq).
    merge_operations -- ordered list of bigrams (tuples of two symbols) that should be merged.
    num_symbols -- how many merge operations to actually perform. Can be less than the number of merge 
    operations that were passed.
    use_eol -- whether to use eol symbol when performing segmentation (see Nematus BPE paper).

    Returns:
    segmentations -- final segmentations for words in vocab using BPE.
    """
    segmentations = {}
    quick_pairs = defaultdict(lambda: set())
    
    for word in vocab:
        #Set up segmentations data structure
        seg = list(word)
        if use_eol:
            seg.append("</w>")
        segmentations[word] = seg
        #Set up the quick_find data structure
        for idx, c in enumerate(seg):
            #Now set up the quick_pairs data structure
            if idx != len(seg) - 1:
                quick_pairs[(c, seg[idx+1])].add((word, idx, idx+1))
    
    #Only do the first n merge operations...
    if num_symbols != None:
        merge_operations = merge_operations[:int(num_symbols)]
    
    for pair in merge_operations:
        new_symbol = "".join(pair)
        #Some of the pairs aren't relevant to the evaluations set...
        if pair in quick_pairs:
            involved_words = quick_pairs[pair]

            while involved_words:
                word, first_index, second_index = involved_words.pop()
                #Call this with throw away dicts for the frequencey cache and all_freqs. Not relevant here at all.
                core_word_update(vocab, word, pair, new_symbol, first_index, second_index, quick_pairs, segmentations, Counter(), Counter(), False)
            quick_pairs.pop(pair)
            
    return segmentations


#Take segmentations for a corpus and then split the corpus file itself
def delimit_corpus(corpus_path, output_path, segmentations, separator ="@@"):
    """
    Not part of the main BPE pipeline. Takes a corpus file as input, as well as segmentations for the 
    words in that corpus. Segments the input corpus using the segmentations given.

    Arguments:
    corpus_path -- path to the corpus to segment.
    output_path -- path that the segmented corpus should be written to.
    segmentations -- Dict of word -> list of word parts. There should be a segmentation for 
    each word in the input corpus.
    separator -- connector between segments of a word in the segmented corpus.

    Returns:
    None. Writes a file.
    """
    corpus_obj = open(corpus_path, "r")
    output_obj = open(output_path, "w+")
    for line in corpus_obj:
        parsed_contents = []
        for word in line.split():
            word_seg = segmentations[word]
            for part in word_seg[:-1]:
                parsed_contents.append(part + separator)
            parsed_contents.append(word_seg[-1])

        parsed_line = " ".join(parsed_contents)
        output_obj.write(parsed_line.strip())
        output_obj.write('\n')
    
    corpus_obj.close()
    output_obj.close()


def segment_vocab(vocab, num_iterations, use_eol=False, valid_freq=None, valid_func=None): 
    """
    The heart and soul of BPE that actually takes an input vocab in and segments it. 

    Arguments:
    vocab -- Dict of (word -> freq). Usually extracted from corpus file and then passed in.
    num_iterations -- number of merge operations that should be performed.
    use_eol -- Whether to use an eol symbol during segmentation (see Nematus BPE paper)
    valid_freq -- How often segmentations should be evaluated, in terms of iterations.
    valid_func -- Function that should be used for segmentation evaluation. 
    (Note that expected form for a validation function is f(segs, op_number))

    Returns:
    segmentations -- final segmentations for words in vocab using BPE.
    merges_done -- list of bigrams that were merged (in order) by BPE.
    val_scores -- ONLY returned if validation function and freq were passed in. Scores for 
    segmentations, taken every valid_freq number of merge operations.
    """
    val_scores = []
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
        if use_eol:
            seg.append("</w>")
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
        if valid_freq != None and i%valid_freq == 0:
            #Note that expected form for a validation function is f(segs, op_number)
            cur_score = valid_func(segmentations, i)
            val_scores.append(cur_score)
        #Has the max frequencey cache gone dry?
        threshold = check_cache(freq_cache, threshold, all_freqs, i)

        drawn_pairs = draw_frequent_pairs(freq_cache)            
        best_pair = drawn_pairs[0]

        sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, best_pair[0], best_pair[1], freq_cache[best_pair]))                           
        
        merge_update(vocab, best_pair, quick_pairs, quick_find, segmentations, freq_cache, all_freqs, threshold)
        merges_done.append(best_pair)

    if valid_freq != None:
        #Get one last validation on the max number of segs
        cur_score = valid_func(segmentations, num_iterations)
        val_scores.append(cur_score)
        return segmentations, merges_done, val_scores

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
    write_segmentation_list(args.output, vocab, segmentations)

    #Write out the merge operations in the order they were done.
    merge_ops_output_obj = open(args.output + "_merge_ops.txt", "w+")
    for pair in merges_done:
        merge_ops_output_obj.write(" ".join(pair))
        merge_ops_output_obj.write('\n')
    merge_ops_output_obj.close()













        




