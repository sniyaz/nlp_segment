import sys
import pickle
import os
sys.path.append("../")
from bpe import apply_merge_ops, delimit_corpus, get_vocabulary, apply_presegs, segment_vocab, recover_preseg_boundary, write_segmentation_list, remove_eols
from copy import deepcopy
from collections import Counter

import pdb

"""Compare our segmentations with those from a BPE implementation"""

if __name__ == '__main__':

    #Absolutely horrific coding style that makes this an apples to apples comparison.
    PYTHONHASHSEED = 0

    corpus_file = sys.argv[1]
    presegs_file = sys.argv[2]
    output_name = sys.argv[3]

    #Load in presegs
    presegs = open(presegs_file, "rb")
    presegs = pickle.load(presegs)
    
    #Apply the presegs to morpho_file
    corpus = open(corpus_file, "r")
    morpho_vocab = get_vocabulary(corpus)
    morpho_preseg = apply_presegs(morpho_vocab, presegs)

    #TODO: Avoid hard-coding
    num_iters = 32000
    #Segment morpho version
    morpho_segmentations, morpho_operations = segment_vocab(morpho_preseg, num_iters, use_eol=False)

    #Recover segmentations of training corpora and write then out
    final_morpho_seg = recover_preseg_boundary(morpho_vocab, presegs, morpho_segmentations)
    write_segmentation_list(output_name + "_morpho", morpho_vocab, final_morpho_seg)

    #Now...do the baseline!
    base_vocab = get_vocabulary(corpus)
    corpus.close()
    base_segmentations, base_operations = segment_vocab(base_vocab, num_iters, use_eol=False)
    write_segmentation_list(output_name + "_base", base_vocab, base_segmentations)

   