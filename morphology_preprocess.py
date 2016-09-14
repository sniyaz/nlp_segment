import json
import pickle
from collections import defaultdict
from bpe import get_vocabulary, cosine_similarity 
import networkx as nx 
import sys
import os

sys.path.append("./evaluation/")
from evaluate_seg import get_gs_data

import pdb

def process_json(json_data):
    morph_transforms = defaultdict( lambda: defaultdict(lambda: []) )
    
    for change in json_data:
        #Don't care about empty rules like this...
        if "transformations" not in change:
            continue

        change_data = {}
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
        if rule_inverted:
            direction_words = set([(trans["direction"]["output"], trans["direction"]["input"]) for trans in transformations])
        else:
            direction_words = set([(trans["direction"]["input"], trans["direction"]["output"]) for trans in transformations])
        
        direction_words = list(direction_words)
        direction_words = [(to_str,) + tup for tup in direction_words]

        morph_transforms[from_str][rule_kind].extend(direction_words)
    
    #Heuristic: Try transforms that lead to shorter words first.
    for drop_string in morph_transforms:
        morph_transforms[drop_string]["s"] = sorted(morph_transforms[drop_string]["s"], key=lambda x: len(x[0]))
        morph_transforms[drop_string]["p"] = sorted(morph_transforms[drop_string]["p"], key=lambda x: len(x[0]))

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
    while target_words:
        #rint(i)
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
            propogate_to_children(propogation_graph, presegs, word, 0, drop_str, change_kind)

            if test_set:
                target_words.append(parent_word)
          
        
        i += 1 
       
    #DEBUG
    to_write = sorted(list(presegs.keys()), key=lambda x: len(x), reverse=True)
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
    for child in graph.successors(word):
        link = graph[word][child]["link"]
        idx = link[prev_idx]
        segment = presegs[child][idx]
        if kind == "s":
            if segment[-len(drop_str):] == drop_str and len(segment) > len(drop_str):
                presegs[child][idx] = segment[:-len(drop_str)]
                presegs[child].insert(idx + 1, drop_str)
                link.append(max(link) + 1)
                propogate_to_children(graph, presegs, child, idx, drop_str, kind)
    
        else:
            if segment[:len(drop_str)] == drop_str and len(segment) > len(drop_str):
                presegs[child][idx] = drop_str
                presegs[child].insert(idx + 1, segment[len(drop_str):])
                link.append(max(link) + 1)
                propogate_to_children(graph, presegs, child, idx, drop_str, kind)


def test_transforms(word, morph_transforms, vocab, word_vectors):
    threshold = 0.3
    
    for i in range(1, len(word)):
        suffix = word[-i:]
        prefix = word[:i]

        for kind, direction_words in zip(["s", "p"], [morph_transforms[suffix]["s"], morph_transforms[prefix]["p"]]):
            for transform in direction_words:                
                to_str = transform[0]

                if kind == "s":
                    new_string = word[:-i] + to_str
                else:
                    new_string = to_str + word[i:]
                trans_from = transform[1]
                trans_to = transform[2]

                if new_string not in vocab or trans_from not in vocab or trans_to not in vocab:
                    continue

                direction_vector = word_vectors[trans_to] - word_vectors[trans_from]
                new_vector = word_vectors[word] + direction_vector
                #print(cosine_similarity(word_vectors[new_string], new_vector))
                if cosine_similarity(word_vectors[new_string], new_vector) > threshold:
                    print(word + " ---> " + new_string)
                    if kind == "s":
                        return "s", suffix, new_string 
                    else:
                        return "p", prefix, new_string
        
   

if __name__ == '__main__':

    vocab = get_vocabulary(open("data/europarl/fi_san.txt", "r"))  
        
    test = json.load(open("data/morph_rules.json", "r"))
    morph_transforms = process_json(test)    
    
    word_vectors = pickle.load(open("/Users/Sherdil/Research/NLP/nlp_segment/data/vectors.txt", "rb"))

    #If prepping experiment
    gold_standard = {}
    eval_order = []
    get_gs_data(open("/Users/Sherdil/Research/NLP/nlp_segment/data/seg_eval/goldstdsample.eng.txt", "r"), gold_standard, eval_order) 
    pdb.set_trace()  
    compute_preseg(vocab, word_vectors, morph_transforms, test_set=list(gold_standard.keys()))

    #compute_preseg(vocab, word_vectors, morph_transforms)

