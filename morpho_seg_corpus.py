"""
Uses the pre-segmentation and BPE alorithms to break up training and source corpora for the 
POS tagging task.
"""

import sys
import pickle
import os
from bpe import apply_merge_ops, delimit_corpus, get_vocabulary, apply_presegs, segment_vocab, recover_preseg_boundary, remove_eols
from copy import deepcopy
from collections import Counter

import pdb

if __name__ == '__main__':

    training_file = sys.argv[1]
    val_file = sys.argv[2]

    presegs_file = sys.argv[3]
    save_name = sys.argv[4]
    num_iters = int(sys.argv[5])
    #Bit for using eol stuff.
    ignore_case = int(sys.argv[6])

    #Load in presegs.
    presegs = open(presegs_file, "rb")
    presegs = pickle.load(presegs)

    #PHASE 1: THE TRAINING CORPORA
    
    #Apply the presegs to the training corpora
    training_obj = open(training_file, "r")
    training_vocab = get_vocabulary(training_obj, ignore_case=ignore_case)
    training_obj.close()
    training_preseg_vocab = apply_presegs(training_vocab, presegs)

    #Train BPE
    _, merge_operations = segment_vocab(training_preseg_vocab, num_iters)

    #pdb.set_trace()  

    #PHASE 2: THE VALIDATION CORPORA

    #Apply the presegs to the validation corpora
    val_obj = open(val_file, "r")
    val_vocab = get_vocabulary(val_obj, ignore_case=ignore_case)
    val_obj.close()
    val_preseg_vocab = apply_presegs(val_vocab, presegs)
   
    #Apply trained BPE operations to validation corpora
    val_intermediate_seg = apply_merge_ops(val_preseg_vocab, merge_operations)

    #Recover final segmentations of validation corpora and write them out.
    final_val_seg = recover_preseg_boundary(val_vocab, presegs, val_intermediate_seg)
    delimit_corpus(val_file, save_name, final_val_seg, restore_case=ignore_case)

     