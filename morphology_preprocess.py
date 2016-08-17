import json
import pickle
from collections import defaultdict
from bpe import get_vocabulary, cosine_similarity 

import pdb


def process_json(json_data):
    prefix_transforms = defaultdict(lambda: [])
    suffix_transforms = defaultdict(lambda: [])
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
        drop_string = get_drop_string(from_str, to_str, rule_kind)
 
	transformations = sorted(change["transformations"], key=lambda x: float(x["cosine"]), reverse=True)
	if rule_inverted:
            transformations = [(trans["rule"]["output"], trans["rule"]["input"]) for trans in transformations]
        else:
            transformations = [(trans["rule"]["input"], trans["rule"]["output"]) for trans in transformations] 

        change_data["from"] = from_str
        change_data["to"] = to_str
        change_data["kind"] = rule_kind
        change_data["drop_str"] = drop_string
        change_data["transforms"] = transformations

        if rule_kind == "s":
            suffix_transforms[drop_string].append(change_data)
        else:
           prefix_transforms[drop_string].append(change_data)

	#if rule_kind  == "p":
	    #pdb.set_trace()
    
    #pdb.set_trace()
    for suffix in suffix_transforms:
        suffix_transforms[suffix] = sorted(suffix_transforms[suffix], key=lambda x: len(x["to"]))
    for prefix in prefix_transforms:
        prefix_transforms[prefix]= sorted(prefix_transforms[prefix], key=lambda x: len(x["to"]))
    return prefix_transforms, suffix_transforms


def get_drop_string(from_str, to_str, rule_kind):
    chars = [char for char in to_str]
    if rule_kind == "p":
        chars.reverse()
    for char in chars:
        if rule_kind == "s":
	    if from_str[0] == char:
	        from_str = from_str[1:]
	    else:
	        break
	else:
	    if from_str[-1] == char:
	        from_str = from_str[:-1]
	    else:
	        break
    
    return from_str
        
      
      
def compute_preseg(vocabulary, word_vectors, prefix_transforms, suffix_transforms):
    threshold = 0.5

    vocab = list(vocabulary.keys())
    #Go from longer words to shorter ones since we apply "drop" rules
    vocab = sorted(vocab, key = lambda x: len(x), reverse=True)
    presegs = {}
    i = 0
    for word in vocab:
        #print(i)
        possible_change = test_transforms(word, prefix_transforms, suffix_transforms, vocab, word_vectors)
	if possible_change:
	    change_kind, drop_str = possible_change
	    if change_kind == "s":
	        presegs[word] = [word[:-len(drop_str)], drop_str]
	    else:
	        presegs[word] = [drop_str, word[len(drop_str):]]
	    print(presegs[word])

        i += 1 
   
   
    vocab.sort()
    segs_output_obj = open("debug_temp/" + "pre_segs.txt", "w+")
    for word in vocab:
        final_seg = presegs[word]
        delimited_seg = " ".join(final_seg)
        segs_output_obj.write(word + ": " + delimited_seg)
        segs_output_obj.write('\n')
    segs_output_obj.close()
	        

def test_transforms(word, prefix_transforms, suffix_transforms, vocab, word_vectors):
    threshold = 0.5
    
    for i in range(1, len(word)):
	suffix = word[-i:]
        prefix = word[:i]
        possible_changes = suffix_transforms[suffix] + prefix_transforms[prefix]
        for change_data in possible_changes:
            from_str = change_data["from"]
    	    to_str = change_data["to"]
	    

	    if change_data["kind"] == "s":
                #Gotta check that you can actually apply the transform...
	        if word[-len(from_str):] != from_str:
	            continue
		new_string = word[:-len(from_str)] + to_str
	    else:
                #Gotta check that you can actually apply the transform...
	        if word[:len(from_str)] != from_str:
	            continue
		new_string = to_str + word[len(from_str):]
	    
	    if new_string not in vocab:
    	        continue
    	    
	    for transform in change_data["transforms"]:
    	        trans_from = transform[0]
    	        trans_to = transform[1]
    	        if trans_from not in vocab or trans_to not in vocab:
    	            continue
    	        direction_vector = word_vectors[trans_to] - word_vectors[trans_from]
    	        new_vector = word_vectors[word] + direction_vector
    	        if cosine_similarity(word_vectors[new_string], new_vector) > threshold:
		    #pdb.set_trace()
		    print(word + " ---> " + new_string)
		    #print("from: " + change_data["from"])
		    #print("to: " + change_data["to"])
		    #print("drop: " + change_data["drop_str"])
		    return change_data["kind"], change_data["drop_str"]
    
    
   


if __name__ == '__main__':
    
    vocab = get_vocabulary(open("data/europarl/fi_san.txt", "r"))     
    test = json.load(open("data/morph_rules.json", "r"))
    prefix_transforms, suffix_transforms = process_json(test)
    
    #prefix_transforms, suffix_transforms = pickle.load(open("data/processed_rules.txt", "rb")) 
    #f_obj = open("data/processed_rules.txt", "wb+")
    #pickle.dump((dict(prefix_transforms), dict(suffix_transforms)), f_obj)
    #f_obj.close()
    
    
    word_vectors = pickle.load(open("/Users/Sherdil/Research/NLP/nlp_segment/data/vectors.txt", "rb"))
    
    compute_preseg(vocab, word_vectors, prefix_transforms, suffix_transforms)