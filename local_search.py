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
import zlib

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
        help="1 -> Vanilla BPE, 2-> Tie Breaking BPE, 3-> Local Optimization")
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
    parser.add_argument(
        '--gamma', '-g', action="store", required=False,
        help="Value of hyperparameter if doing local optimization")
    
    return parser


def get_vocabulary(fobj):
    fobj.seek(0)
    vocab = Counter()
    for line in fobj:
        line = line.strip()
        for word in line.split():
            vocab[word] += 1

    return vocab


def get_vocabulary_freq_table(fobj):
    fobj.seek(0)
    #Write out a pure corpus so we know what we trained on. "Pure" means it was in the 300k word vectors.
    pure_corpus_obj = open(args.output + "_pure_corpus.txt", "w+")
    vocab = Counter()
    missed = 0
    for line in fobj:
        original_line = line
        line = line.strip()
        line_parts = line.split()
        freq = int(line_parts[0])
        word = line_parts[1]
        if word in word_vectors:
            vocab[word] += freq
            pure_corpus_obj.write(original_line)
        else:
            missed += 1
            
    print("MISSED: ")
    print(missed)
    return vocab


#Needed if doing BPE with tie breaking. Big table of all pair frequencies
def get_pair_statistics():
    all_freqs = Counter()
    for word, freq in vocab.items():
        prev_char = word[0]
        for char in word[1:]:
            all_freqs[(prev_char, char)] += freq
            prev_char = char

    return all_freqs


def get_similarity(word1, word2):
    return np.dot(word_vectors[word1], word_vectors[word2])


#Euclidean distance between 2 vectors
def distance(vec1, vec2):
    difference = vec1 - vec2
    return np.linalg.norm(difference)


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


def get_mean(sample_set):
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


def get_set_cohesion(word_set):
    involved_words = [i[0] for i in word_set]
    pair_words = set()
    for word in involved_words:
        pair_words.add((word,))
    if len(pair_words) == 0:
        return 0
    #Center of the new symbol's vectors
    pair_mean = get_mean(pair_words)
    average_similarity = 0

    for word in pair_words:
        average_similarity += np.dot(word_vectors[word[0]], pair_mean)
    
    #pdb.set_trace()
    average_similarity = average_similarity/len(pair_words)
   
    return average_similarity


def delta_cache_usable(pair):
    delta_cache_info = delta_cache[pair]
    timestamp = delta_cache_info[1]
    for char in pair:
        if char in char_change_log and char_change_log[char] > timestamp:
            return False
    return True


def get_pair_delta(pair):
    if pair in delta_cache and delta_cache_usable(pair):
        #print("SAVED!")
        return delta_cache[pair][0]
    length_delta = -all_freqs[pair]
    old_spread = sigma_cache[pair[0]] + sigma_cache[pair[1]]
    new_spread = 0
    for char in pair:
        new_char_qf = copy.deepcopy(quick_find[char])
        for pair_info in quick_pairs[pair]:
            word = pair_info[0]
            if (word,) in new_char_qf:
                new_char_qf.remove((word,))
        new_spread += get_set_cohesion(new_char_qf)
    
    new_spread += get_set_cohesion(quick_pairs[pair])
    spread_delta = old_spread - new_spread

    total_delta = length_delta + gamma*spread_delta
    delta_cache[pair] = (total_delta, i)
    return total_delta


def get_tuning_cohesion(word, candidate_seg):
    cur_cohesion = 0
    counted_symbols = 0
    for symbol in candidate_seg:
        if symbol in mean_cache:
            cur_cohesion += np.dot(word_vectors[word], mean_cache[symbol][0])
            counted_symbols += 1
    if counted_symbols == 0:
        return 0
    cur_cohesion = cur_cohesion/counted_symbols
    return cur_cohesion

    
#Updates the quick_pairs and segmentations data structures for a given word.
def core_word_update(word, pair, new_symbol, first_index, second_index, quick_pairs, \
    segmentations, freq_changes, update_caches):
    #Delete old info from the pairs data structure (from pairs on a boundary with the new symbol) 
    if second_index + 1 < len(segmentations[word]):
        quick_pairs[(pair[1], segmentations[word][second_index + 1])].remove((word, second_index, second_index + 1))
        if update_caches:
            all_freqs[(pair[1], segmentations[word][second_index + 1])] -= vocab[word]
            freq_changes[(pair[1], segmentations[word][second_index + 1])] = all_freqs[(pair[1], segmentations[word][second_index + 1])]
            if (pair[1], segmentations[word][second_index + 1]) in delta_cache:
                delta_cache.pop((pair[1], segmentations[word][second_index + 1]))

    if first_index - 1 >= 0: 
        quick_pairs[(segmentations[word][first_index - 1], pair[0])].remove((word, first_index - 1, first_index))
        if update_caches:
            all_freqs[(segmentations[word][first_index - 1], pair[0])] -= vocab[word]
            freq_changes[(segmentations[word][first_index - 1], pair[0])] = all_freqs[(segmentations[word][first_index - 1], pair[0])]
            if (segmentations[word][first_index - 1], pair[0]) in delta_cache:
                delta_cache.pop((segmentations[word][first_index - 1], pair[0]))

    
    #Update segmentations data structure 
    segmentations[word][first_index] = new_symbol
    segmentations[word].pop(second_index)

    #Update the pairs data structure with new pairs formed with new symbol 
    if second_index < len(segmentations[word]):
        if (new_symbol, segmentations[word][second_index]) not in quick_pairs:                
            quick_pairs[(new_symbol, segmentations[word][second_index])] = set([(word, first_index, second_index)])
        else:
            quick_pairs[(new_symbol, segmentations[word][second_index])].add((word, first_index, second_index))
        if update_caches:
            all_freqs[(new_symbol, segmentations[word][second_index])] += vocab[word]
            freq_changes[(new_symbol, segmentations[word][second_index])] += vocab[word]
        
    if first_index - 1 >= 0:
        if (segmentations[word][first_index - 1], new_symbol) not in quick_pairs:
            quick_pairs[(segmentations[word][first_index - 1], new_symbol)] = set([(word, first_index - 1 , first_index)])
        else:
            quick_pairs[(segmentations[word][first_index -1], new_symbol)].add((word, first_index - 1 , first_index))
        if update_caches: 
            all_freqs[(segmentations[word][first_index - 1], new_symbol)] += vocab[word]
            freq_changes[(segmentations[word][first_index - 1], new_symbol)] += vocab[word]
    
    #Now, move the indicies for things after the merged pair!
    for i in range(second_index, len(segmentations[word]) - 1):
        quick_pairs[(segmentations[word][i], segmentations[word][i+1])].remove((word, i + 1 , i + 2))
        quick_pairs[(segmentations[word][i], segmentations[word][i+1])].add((word, i , i + 1))
        

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

    #Book keeping if doing BPE tie breaking..
    freq_changes = Counter()
    #Edge cases can have you change the set as you iterate over it!
    while involved_words:
        word, first_index, second_index = involved_words.pop()

        core_word_update(word, pair, new_symbol, first_index, second_index, quick_pairs, segmentations, freq_changes, True)
        affected_words.add(word)

        #Remove the mapping of the word and old symbols from the quick_find structure
        if remove_word_check(word, pair[0]):
            quick_find[pair[0]].remove((word,))
            char_change_log[pair[0]] = i
        if remove_word_check(word, pair[1]):
            #New edge case in situations like "l" + "l"
            if pair[0] != pair[1]:
                quick_find[pair[1]].remove((word,))
                char_change_log[pair[1]] = i
        #Update q_find data structure with new symbol
        if new_symbol not in quick_find:
            quick_find[new_symbol] = set([(word,)])
        else:
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


def deterministic_hash(data):
    data = str(data).encode('utf-8')
    return zlib.adler32(data)


def draw_random_pairs():
    pairs = list(quick_pairs.keys())
    drawn_indicies = [randint(0, len(pairs) - 1) for i in range(search_scatter)]
    drawn_pairs = [pairs[i] for i in drawn_indicies]
    return drawn_pairs


#Check the freq_cache and refresh if needed.
def check_cache():
    if len(freq_cache) == 0:
        most_frequent = max(all_freqs, key=all_freqs.get)
        if i == 0:
            refresh_freq_cache(all_freqs[most_frequent]/10)
        else:
            refresh_freq_cache(all_freqs[most_frequent] * i/(i+10000.0))    
    
    
def refresh_freq_cache(new_threshold):
    global threshold
    freq_cache.clear()
    threshold = new_threshold
    for pair in all_freqs:
        if all_freqs[pair] > threshold:
            freq_cache[pair] = all_freqs[pair]


#Inspired by the BPE Paper code. Remember to cite if needed.
def draw_frequent_pairs():
    #Repopulate the cache in the event that it goes dry
    check_cache()
        
    #print("SIZE CACHE:")
    #print(len(freq_cache))
    frequent_pair = max(freq_cache, key=freq_cache.get)
    most_frequent_pairs = [p for p in freq_cache if freq_cache[p] == freq_cache[frequent_pair]]

    #To prevent non-determinism arising from the dictionaries
    return sorted(most_frequent_pairs, key=deterministic_hash)



def update_sigma_cache(pair):
    for char in pair:
        sigma_cache[char] = get_set_cohesion(quick_find[char])
    new_symbol = "".join(pair)
    sigma_cache[new_symbol] = get_set_cohesion(quick_find[new_symbol])


#If we manually get the merge that reduces our objective most. Local optimization stratagey!
def get_next_state():
    check_cache()
    pruned = False
    best_pair_so_far = None
    best_drop_so_far = float("-inf")
    seen = set()

    spread_calcs_done = 0

    while not pruned:
        pair_list = list(freq_cache.keys())
        pair_list = sorted(pair_list, key=freq_cache.get, reverse=True)
        for pair in pair_list:
            if pair in seen:
                continue
            elif all_freqs[pair] + gamma*3 < best_drop_so_far:
                pruned = True
                break
            #Even if we can't stop completely, there's no point looking at a pair like this. Prune just this one.
            elif all_freqs[pair] + gamma*(3 - (sigma_cache[pair[0]] + sigma_cache[pair[1]])) < best_drop_so_far:
                continue
            else:
                cur_drop = -get_pair_delta(pair)
                spread_calcs_done += 1
                if cur_drop > best_drop_so_far:
                    best_drop_so_far = cur_drop
                    best_pair_so_far = pair
            seen.add(pair)
        #If we didn't hit the cuttoff point, then we need to expand the cache and keep trying...
        if not pruned:
            refresh_freq_cache(0.5*threshold)
            #print("REFRESHING")

    #print("SPREAD CALCS DONE: " + str(spread_calcs_done))
    #print("BEST DROP: " + str(best_drop_so_far))
    return best_pair_so_far


def use_tuned_seg(word):
    if word == "nightstand":
        pdb.set_trace()
    old_seg = segmentations[word]
    candidates = [seg for seg in get_segmentations(word) if len(seg) == len(old_seg)]
    next_seg = max(candidates, key=lambda x: get_tuning_cohesion(word, x))
    for symbol in set(old_seg):
        if symbol not in next_seg:
            quick_find[symbol].remove((word,))
            new_count = mean_cache[symbol][1] - 1
            if new_count == 0:
                mean_cache.pop(symbol)
            else:
                total_mass = mean_cache[symbol][0]*mean_cache[symbol][1]
                total_mass -= word_vectors[word]
                new_average = total_mass/new_count
                mean_cache[symbol] = (new_average, new_count)
    for symbol in set(next_seg):
        if symbol not in old_seg:
            quick_find[symbol].add((word,))
            if symbol not in mean_cache:
                mean_cache[symbol] = (word_vectors[word], 1)
            else:
                new_count = mean_cache[symbol][1] + 1
                total_mass = mean_cache[symbol][0]*mean_cache[symbol][1]
                total_mass += word_vectors[word]
                new_average = total_mass/new_count
                mean_cache[symbol] = (new_average, new_count)
    
    
    prev_char = old_seg[0]
    idx = 0
    for char in old_seg[1:]:
        all_freqs[(prev_char, char)] -= vocab[word]
        quick_pairs[(prev_char, char)].remove((word, idx, idx + 1)) 
        prev_char = char
        idx += 1
    
    prev_char = next_seg[0]
    idx = 0
    for char in next_seg[1:]:
        all_freqs[(prev_char, char)] += vocab[word]
        quick_pairs[(prev_char, char)].add((word, idx, idx + 1)) 
        prev_char = char
        idx += 1 

    if word == "nightstand":
        pdb.set_trace()

    segmentations[word] = next_seg
    


if __name__ == '__main__':
    
    #Main hyperparameters!
    sample_size = 100
    #Only if employing as more than just tie-breaker
    search_scatter = 100
    
    parser = create_parser()
    args = parser.parse_args()

    mode = int(args.mode)
    if mode == 1: 
        use_bpe = True
        tie_break_only = True
    elif mode == 2:
        use_bpe = False
        tie_break_only = True
    elif mode == 3:
        use_bpe = False
        tie_break_only = False
        gamma = float(args.gamma)
    elif mode == 4:
        use_bpe = True
        tie_break_only = True
        mix_intervals = 1000

    word_vectors = pickle.load(open("/Users/Sherdil/Research/NLP/nlp_segment/data/vectors.txt", "rb"))
    segmentations = {}
    
    #Dict that maps characters to words containing them
    quick_find = defaultdict(lambda: set())
    #Dict that maps pairs to words containing them
    quick_pairs = defaultdict(lambda: set())
    #Cache for speeing up max when finding frequent pairs
    threshold = None
    freq_cache = Counter()
    if args.ft:
        vocab = get_vocabulary_freq_table(args.input)
    else:
        vocab = get_vocabulary(args.input)
    all_freqs = get_pair_statistics()


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

    #Fill the sigma cache to start thigns off.
    sigma_cache = {}
    for char in quick_find:
        sigma_cache[char] = get_set_cohesion(quick_find[char])

    char_change_log = {}
    delta_cache = {}


    print("SIZE QUICK FIND")
    print(len(quick_find.keys()))

    print("SIZE QUICK PAIRS")
    print(len(quick_pairs.keys()))

    print("SIZE VOCAB")
    print(len(vocab.keys()))

    num_ties = 0
    num_iterations = int(args.symbols)
    merges_done = []
    affected_words = set()

    #Core algorithm
    for i in range(num_iterations):
        
        #If we tune/mix the segmentations every so often
        if mode == 4:
            if i != 0 and i % mix_intervals == 0:
                to_tune = list(affected_words)
                to_tune = sorted(to_tune, key=vocab.get)
                mean_cache = {}
                for char in quick_find:
                    if len(quick_find[char]) > 0:
                        mean_cache[char] = (get_mean(quick_find[char]), len(quick_find[char]))
                j = 0
                for word in to_tune:
                    old_seg = segmentations[word]
                    use_tuned_seg(word)
                    new_seg = segmentations[word]
                    print("tuning " + str(j) + " of " + str(len(to_tune)) + ": "  + str(old_seg) + " --> " + str(new_seg))
                    j += 1

                refresh_freq_cache(threshold)
                affected_words = set()
        
        #Look at many merges, and then pick the best one
        if tie_break_only:
            drawn_pairs = draw_frequent_pairs()            
            if use_bpe or len(drawn_pairs) == 1:
                best_pair = drawn_pairs[0]
            else:
                num_ties += 1
                best_pair = max(drawn_pairs, key=lambda x: get_set_cohesion(quick_pairs[x]))
              
        else:
            best_pair = get_next_state()
    
        sys.stderr.write('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, best_pair[0], best_pair[1], freq_cache[best_pair]))                           
        
        merge_update(best_pair)
        merges_done.append(best_pair)

        if not tie_break_only:
            #Now that we've picked a pair, update the sigma_cache for future use.
            update_sigma_cache(best_pair)

    
    
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

    print("NUM TIES WAS:")
    print(num_ties)













        




