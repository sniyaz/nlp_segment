import sys
import pickle
import os
sys.path.append("../")
from bpe import apply_merge_ops, delimit_corpus, get_vocabulary, apply_presegs, segment_vocab, recover_preseg_boundary 
from copy import deepcopy
from collections import Counter

import pdb

"""Take the final segmented training corpora for baseline and morpho and prune out lines greater than 
a length consistantly (for fairness in translation experiment."""

def seq_len(input_line):
    input_line = input_line.strip()
    line_parts = input_line.split()
    return len(line_parts)

if __name__ == '__main__':

    baseline_src = sys.argv[1]
    baseline_target = sys.argv[2]

    morpho_src = sys.argv[3]
    morpho_target = sys.argv[4]

    max_length = int(sys.argv[5])

    pruned_baseline_src = baseline_src + ".pruned"
    pruned_baseline_target = baseline_target + ".pruned"
    pruned_morpho_src = morpho_src + ".pruned"
    pruned_morpho_target = morpho_target + ".pruned"

    baseline_src_obj = open(baseline_src, "r")
    baseline_target_obj = open(baseline_target, "r")
    morpho_src_obj = open(morpho_src, "r")
    morpho_target_obj = open(morpho_target, "r")

    pruned_baseline_src_obj = open(pruned_baseline_src, "w+")
    pruned_baseline_target_obj = open(pruned_baseline_target, "w+")
    pruned_morpho_src_obj = open(pruned_morpho_src, "w+")
    pruned_morpho_target_obj = open(pruned_morpho_target, "w+")

    while True:
        baseline_src_line = baseline_src_obj.readline()
        if baseline_src_line == "":
            break
        baseline_target_line = baseline_target_obj.readline()
        morpho_src_line = morpho_src_obj.readline()
        morpho_target_line = morpho_target_obj.readline()

        if seq_len(baseline_src_line) < max_length and seq_len(baseline_target_line) < max_length and seq_len(morpho_src_line) < max_length and seq_len(morpho_target_line) < max_length:
            pruned_baseline_src_obj.write(baseline_src_line)
            pruned_baseline_target_obj.write(baseline_target_line)
            pruned_morpho_src_obj.write(morpho_src_line)
            pruned_morpho_target_obj.write(morpho_target_line)
    
    baseline_src_obj.close()
    baseline_target_obj.close()
    morpho_src_obj.close()
    morpho_target_obj.close()

    pruned_baseline_src_obj.close()
    pruned_baseline_target_obj.close()
    pruned_morpho_src_obj.close()
    pruned_morpho_target_obj.close()
        

            





    




