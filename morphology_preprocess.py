"""
Algorithm that generates pre-segmentations on a target corpus. See the write up for more on 
how exactly this process works.
"""
import json
import pickle
from collections import defaultdict
from bpe import get_vocabulary, get_vocabulary_freq_table 
import networkx as nx 
import sys
import os
import numpy as np

sys.path.append("./evaluation/")
from evaluate_seg import get_gs_data

import pdb
import time

def process_json(json_contents, vocab, word_vectors, num_examples=float("inf")):
    """
    Extends a list of example morphological transformations to the given corpus/vocab. Does this
    via a set of vector embeddings, which need to be trained before-hand.

    Arguments:
    json_contents -- json object containing the example transformations (format currently defined by the 
    json output from John's Soricut and Och implementation)
    vocab -- Dict of (word -> freq). Extracted from target corpus.
    word_vectors -- Dict of word -> embeddings. Trained on target corpus.
    num_examples -- for some input rule from json file, max number of example direction vectors to store (for speed).

    Returns:
    morph_transforms -- This is a weirdly formatted dictionary (to speed up later code).
    General structure: Rule Kind -> From String -> To String -> Example Direction Vectors. 
    """
    #Satanic code that makes a three-layer dictionary.
    transforms_dict = defaultdict( lambda: defaultdict( lambda: defaultdict(lambda: []) ) )

    for change in json_contents:
        #Don't care about empty rules like this...
        if "transformations" not in change:
            continue

        from_str = change["rule"]["from"]
        to_str = change["rule"]["to"]
        rule_kind = change["rule"]["kind"]
        
        #To prevent infinite loops while propogating. Also, we also want "reductions"
        if len(from_str) == len(to_str):
            continue

        rule_inverted = len(from_str) < len(to_str)
        if rule_inverted:
            from_str, to_str = to_str, from_str

        from_str, to_str = get_drop_string(from_str, to_str, rule_kind)

        transformations = sorted(change["transformations"], key=lambda x: float(x["cosine"]), reverse=True)
        d_vectors = []
        for trans in transformations:
            #OPTIMIZATION: Only keep around a small number of word vectors for each rule.
            if len(d_vectors) == num_examples:
                break

            if rule_inverted:
                longer_word = trans["rule"]["output"]
                shorter_word = trans["rule"]["input"]
            else:
                longer_word = trans["rule"]["input"]
                shorter_word = trans["rule"]["output"]

            if longer_word in vocab and shorter_word in vocab:
                direction_vector = word_vectors[shorter_word] - word_vectors[longer_word]
                d_vectors.append(direction_vector)
            
        transforms_dict[rule_kind][from_str][to_str].extend(d_vectors)
    
    #Build a final list of all transformations.
    morph_transforms = []
    for rule_kind in transforms_dict:
        for from_str in transforms_dict[rule_kind]:
            for to_str in transforms_dict[rule_kind][from_str]:
                d_vectors = tuple(transforms_dict[rule_kind][from_str][to_str])
                morph_transforms.append((rule_kind, from_str, to_str, d_vectors))
    
    #Heuristic: Try transforms that lead to shorter words first.
    morph_transforms = sorted(morph_transforms, key=lambda x: len(x[2]))

    return morph_transforms


def get_drop_string(from_str, to_str, rule_kind):
    """
    Gets drop strings (from string and to string) using the old and new words.

    Arguments:
    from_str -- Longer word in transformation. Will be whittled down.
    to_str -- Shorter word in transformation. Will be whittled down.
    rule_kind -- indicates prefix or suffix.

    Returns:
    from_str -- Smallest substring of longer word that, when removed from longer word, makes 
    the remaining string a substring of the shorter word.
    to_str -- Smallest substring of shorter word that, when removed from shorter word, makes 
    the remaining string a substring of the longer word.
    """
    if rule_kind == "s":
        for char in to_str[:]:
            if from_str[0] == char:
                from_str = from_str[1:]
                to_str = to_str[1:]
            else:
                break
    else:
        for char in to_str[::-1]:
            if from_str[-1] == char:
                from_str = from_str[:-1]
                to_str = to_str[:-1]
            else:
                break
    
    return from_str, to_str
    

def compute_preseg(vocabulary, word_vectors, morph_transforms, test_set=None, threshold=0.5):
    """
    Heart and soul of pre-segmentation algorithm. 

    Arguments:
    vocabulary -- Dict of (word -> freq). Extracted from target corpus.
    word_vectors -- Dict of word -> embeddings. Trained on target corpus.
    morph_transforms -- generalized morphological transformations (from process_json)
    test_set -- If present, a smaller set of words to pre-segment.
    threshold -- minimum cosine similarity needed to trigger a pre-segmentation.

    Returns:
    None. Writes a pickle of the generated pre-segmentations to the save directory give in args,
    as well as a list of pre-segmentations generated.
    """
    propogation_graph = nx.DiGraph()  
    vocab = list(vocabulary.keys())
    #Go from longer words to shorter ones since we apply "drop" rules
    vocab = sorted(vocab, key = lambda x: len(x), reverse=True)
    presegs = {}

    if test_set is None:
        target_words = vocab
    else:
        target_words = sorted(test_set, key = lambda x: len(x), reverse=True)

    i = 0
    words_to_do = len(target_words)
    start = time.time()
    while target_words:
        # if (i % 100 == 0):
        #     print("Processing " + str(i) + "/" + str(words_to_do))
        if (i % 500 == 0):
            print("")
            print("Processing " + str(i) + "/" + str(words_to_do))
            time_elapsed = time.time() - start
            print("Time passed: " + str(time_elapsed))
            print("")
            
        #Don't change something while you iterate over it!
        word = target_words[0]
        target_words = target_words[1:]
        presegs[word] = [word]
        possible_change = test_transforms(word, morph_transforms, vocab, word_vectors, threshold)
        if possible_change:
            change_kind, drop_str, parent_word = possible_change
            if change_kind == "s":
                if spell_change_mode:
                    presegs[word] = [parent_word, drop_str]
                else:
                    presegs[word] = [word[:-len(drop_str)], drop_str]
                propogation_graph.add_edge(parent_word, word, link= [0])
            else:
                if spell_change_mode:
                    presegs[word] = [drop_str, parent_word]
                else:
                    presegs[word] = [drop_str, word[len(drop_str):]]
                propogation_graph.add_edge(parent_word, word, link = [1])

            print(presegs[word])

            #Core of the propogation algorithm!
            if use_propogation:
                propogate_to_children(propogation_graph, presegs, word, 0, drop_str, change_kind)
              
        i += 1
       
    #DEBUG
    to_write = sorted(list(presegs.keys()))
    #return presegs
    
    segs_output_obj = open(save_dir + "pre_segs.txt", "w+")
    for word in to_write:
        final_seg = presegs[word]
        delimited_seg = " ".join(final_seg)
        segs_output_obj.write(word + ": " + delimited_seg)
        segs_output_obj.write('\n')
    segs_output_obj.close()

    with open(os.path.join(save_dir, "presegs_ckpt.txt"), "wb+") as checkpoint_file:
        pickle.dump(presegs, checkpoint_file)
            

def propogate_to_children(graph, presegs, word, prev_idx, drop_str, kind):
    """
    Propogates segmentations of a word to longer words of which it is a base. Uses a graph where
    each word is a node with edges to longer words it is a base of.

    Example: give -> giving.
    """
    try:
        for child in graph.successors(word):
            link = graph[word][child]["link"]
            #BUG FIX: Simtimes things get propagated in one word but not another, meaning link indicies may not line up.
            if len(link) <= prev_idx:
                continue
            idx = link[prev_idx]
            segment = presegs[child][idx]
            if kind == "s":
                if segment[-len(drop_str):] == drop_str and len(segment) > len(drop_str):
                    presegs[child][idx] = segment[:-len(drop_str)]
                    presegs[child].insert(idx + 1, drop_str)
                    print("")
                    print("PROPOGATION!")
                    print(presegs[child])
                    print("")
                    link.append(max(link) + 1)
                    propogate_to_children(graph, presegs, child, idx, drop_str, kind)
        
            else:
                if segment[:len(drop_str)] == drop_str and len(segment) > len(drop_str):
                    presegs[child][idx] = drop_str
                    presegs[child].insert(idx + 1, segment[len(drop_str):])
                    print("")
                    print("PROPOGATION!")
                    print(presegs[child])
                    print("")
                    link.append(max(link) + 1)
                    propogate_to_children(graph, presegs, child, idx, drop_str, kind)
    except Exception as e:
        pdb.set_trace()


def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))


def check_transform_similarity(word, new_string, d_vectors, vocab, word_vectors, threshold):
    """
    Test whether a transformation should be applied to a word, creating a pre-segmentation for that word.

    Arguments:
    word -- word we are testing a transformation on.
    new_string -- word that applying this transformation would create.
    d_vectors -- Example direction vectors for the transformation being tested.
    vocab -- Dict of (word -> freq). Extracted from target corpus.
    word_vectors -- Dict of word -> embeddings. Trained on target corpus.
    threshold -- minimum cosine similarity needed to trigger a pre-segmentation.
    
    Returns:
    Boolean that indicates whether transformation passed the cosine similarity threshold.
    """
    if new_string in vocab:
        canidate_direction_vector = word_vectors[new_string] - word_vectors[word]
        for direction_vector in d_vectors:
            if cosine_similarity(canidate_direction_vector, direction_vector) > threshold:
                print(word + " ---> " + new_string)
                return True
    return False
        

def test_transforms(word, morph_transforms, vocab, word_vectors, threshold):
    """
    For a single word, iterate through all generalized transformations and potentially create a 
    new pre-segmentation for the current word.

    Arguments:
    word -- word we are testing a transformation on.
    morph_transforms -- generalized morphological transformations (from process_json)
    vocab -- Dict of (word -> freq). Extracted from target corpus.
    word_vectors -- Dict of word -> embeddings. Trained on target corpus.
    threshold -- minimum cosine similarity needed to trigger a pre-segmentation.
    
    Returns:
    IF some transformation passed the threshold, return the type (prefix or suffix) that 
    transformation was, the from string (dropped from the input/longer word), and the new word that
    was formed.
    """
    for transform in morph_transforms:
        rule_kind, from_str, to_str, d_vectors = transform
        i = len(from_str)
        if i < len(word):
            if rule_kind == "s":
                if word[-i:] != from_str:
                    continue
                new_string = word[:-i] + to_str
            else:
                if word[:i] != from_str:
                    continue
                new_string = to_str + word[i:]
            if check_transform_similarity(word, new_string, d_vectors, vocab, word_vectors, threshold):   
                return rule_kind, from_str, new_string
    
    

if __name__ == '__main__':

    data_directory = sys.argv[1]
    vectors_file = sys.argv[2]
    transforms_file = sys.argv[3] 
    #Bit that sets whether or not to use propagation algo.
    use_propogation = int(sys.argv[4])
    #Bit if we want to undo spelling changes in segmentations.
    spell_change_mode = int(sys.argv[5])

    threshold = float(sys.argv[6])
    k = int(sys.argv[7])

    save_dir = sys.argv[8]

    word_vectors = pickle.load(open(vectors_file, "rb"))
    json_contents = json.load(open(transforms_file, "r"))

    vocab = get_vocabulary(open(data_directory, "r"))
    test_set = list(vocab.keys())
    #Use Hyperparamter 2 (that blocks presegs on words above a certain freq.)
    test_set = sorted(test_set, key = lambda x: vocab[x], reverse=True)
    test_set = test_set[k:]
       
    morph_transforms = process_json(json_contents, vocab, word_vectors)   
    compute_preseg(vocab, word_vectors, morph_transforms, test_set=test_set, threshold=threshold)
    
