import json
import pickle
from collections import defaultdict
from bpe import get_vocabulary, get_vocabulary_freq_table, cosine_similarity 
import networkx as nx 
import sys
import os

sys.path.append("./evaluation/")
from evaluate_seg import get_gs_data

import pdb
import time

def process_json(json_contents, vocab, word_vectors, num_examples=2):
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
    

def compute_preseg(vocabulary, word_vectors, morph_transforms, test_set=None):

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
        possible_change = test_transforms(word, morph_transforms, vocab, word_vectors)
        if possible_change:
            change_kind, drop_str, parent_word = possible_change
            if change_kind == "s":
                presegs[word] = [word[:-len(drop_str)], drop_str]
                propogation_graph.add_edge(parent_word, word, link= [0])
            else:
                presegs[word] = [drop_str, word[len(drop_str):]]
                propogation_graph.add_edge(parent_word, word, link = [1])
            print(presegs[word])

            #Core of the propogation algorithm!
            if use_propogation:
                propogate_to_children(propogation_graph, presegs, word, 0, drop_str, change_kind)

            if seg_eval:
                target_words.append(parent_word)
              
        i += 1 
       
    #DEBUG
    to_write = sorted(list(presegs.keys()))
    #return presegs
    
    segs_output_obj = open("debug_temp/" + "pre_segs.txt", "w+")
    for word in to_write:
        final_seg = presegs[word]
        delimited_seg = " ".join(final_seg)
        segs_output_obj.write(word + ": " + delimited_seg)
        segs_output_obj.write('\n')
    segs_output_obj.close()

    with open(os.path.join("debug_temp/", "presegs_ckpt.txt"), "wb+") as checkpoint_file:
        pickle.dump(presegs, checkpoint_file)
            

def propogate_to_children(graph, presegs, word, prev_idx, drop_str, kind):
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


def check_transform_similarity(word, new_string, d_vectors, vocab, word_vectors, threshold=0.5):
    if new_string in vocab:
        for direction_vector in d_vectors:
            new_vector = word_vectors[word] + direction_vector
            if cosine_similarity(new_vector, word_vectors[new_string]) > threshold:
                print(word + " ---> " + new_string)
                return True
    return False
        

def test_transforms(word, morph_transforms, vocab, word_vectors):
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
            if check_transform_similarity(word, new_string, d_vectors, vocab, word_vectors):   
                return rule_kind, from_str, new_string
    
    

if __name__ == '__main__':

    data_directory = sys.argv[1]
    vectors_file = sys.argv[2]
    transforms_file = sys.argv[3] 
    #Bit that sets whether or not to use propagation algo.
    use_propogation = int(sys.argv[4])
    #Bit for if we are doing toy seg_eval experiment
    seg_eval = int(sys.argv[5])

    word_vectors = pickle.load(open(vectors_file, "rb"))
    json_contents = json.load(open(transforms_file, "r"))

    if seg_eval:
        #If prepping short "toy" experiment
        corpus_file = os.path.join(data_directory, "pure_corpus.txt")
        vocab = get_vocabulary_freq_table(open(corpus_file, "r"), word_vectors)  
        
        #If prepping short experiment
        gold_standard = {}
        eval_order = []
        gs_file = os.path.join(data_directory, "gs_corpus_only.txt")
        get_gs_data(open(gs_file, "r"), gold_standard, eval_order) 

        test_set = list(gold_standard.keys())

    else:
        vocab = get_vocabulary(open(data_directory, "r"))
        test_set = list(vocab.keys())
        
    morph_transforms = process_json(json_contents, vocab, word_vectors)   
    compute_preseg(vocab, word_vectors, morph_transforms, test_set=test_set)
