import sys
import os
import codecs
import string
import pickle
import json
import numpy as np
import copy
import csv
from collections import defaultdict, Counter

sys.path.append("../")
from bpe import get_vocabulary, apply_presegs, recover_preseg_boundary, apply_merge_ops, segment_vocab

import pdb


if __name__ == '__main__':
    target_corpus = sys.argv[1]
    presegs_file = sys.argv[2]
    output_file = sys.argv[3]
    first_k = int(sys.argv[4])
    num_merges = int(sys.argv[5])

    vocab = get_vocabulary(open(target_corpus, "r"))
    presegs_obj = open(presegs_file, "rb")
    presegs = pickle.load(presegs_obj)
    #Only get the words that were split up differently.
    presegs = {k:presegs[k] for k in presegs if len(presegs[k]) > 1}
    most_freq = list(presegs.keys())
    most_freq = sorted(most_freq, key=lambda x: vocab[x], reverse=True)
    most_freq = most_freq[:first_k]

    #Run some merges and get some date points
    preseg_vocab = apply_presegs(copy.deepcopy(vocab), presegs)
    bpe_segs, _ = segment_vocab(vocab, num_merges)
    morpho_segs, _ = segment_vocab(preseg_vocab, num_merges)
    morpho_segs = recover_preseg_boundary(vocab, presegs, morpho_segs)
    
    output_obj = open(output_file, "w+")
    output_obj.write("WORD PRESEG BPE_SEG MORPHO_SEG FREQ\n")
    for word in most_freq:
        output_obj.write(word + " " + str(presegs[word]) + " " + str(bpe_segs[word]) +  " " + str(morpho_segs[word]) + " " + str(vocab[word]))
        output_obj.write("\n")
        
        







