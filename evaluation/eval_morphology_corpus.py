import sys
import os
import codecs
import argparse
import string
import pickle
import json
import numpy as np
import copy
import csv
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

from evaluate_seg import get_gs_data, call_evaluation, extract_fmeasure, get_merge_ops_list
sys.path.append("../")
from bpe import get_vocabulary, apply_presegs, recover_preseg_boundary, apply_merge_ops, segment_vocab

import pdb

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--corpus', '-c', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input corpus for evaluation")
    parser.add_argument(
        '--gold_standard', '-g', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Gold standard segmentations")
    parser.add_argument(
        '--pre_segs', '-ps', type=argparse.FileType('rb'), default=sys.stdin,
        metavar='PATH',
        help="Pre-seg checkpoint file")
    parser.add_argument(
        '--output', '-o', action="store",
        metavar='PATH',
        help="Output dir")
  
    return parser


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()
    max_merges = 50000
    granularity = 500

    #Extract the gold standard- we'll use it later.
    gold_standard = {}
    eval_order = []
    get_gs_data(args.gold_standard, gold_standard, eval_order)
    gold_standard_file = args.gold_standard.name

    presegs = pickle.load(args.pre_segs)
 
    vocab = get_vocabulary(args.corpus) 
    preseg_vocab = apply_presegs(copy.deepcopy(vocab), presegs)

    _, base_operations = segment_vocab(vocab, max_merges)
    _, morpho_operations = segment_vocab(preseg_vocab, max_merges)

    num_symbols = []
    bpe_scores = []
    morph_scores = []
    
    num_merges = 0
    while (num_merges <= max_merges):
        save_folder = os.path.join(args.output, str(num_merges) + "_merges")
        os.system("mkdir " + save_folder)
        bpe_folder = os.path.join(save_folder, "bpe_folder")
        os.system("mkdir " + bpe_folder)
        morph_folder = os.path.join(save_folder, "morph_folder")
        os.system("mkdir " + morph_folder)
        bpe_eval_file = os.path.join(bpe_folder, "eval_output.txt")
        morph_eval_file = os.path.join(morph_folder, "eval_output.txt")

        base_segs = apply_merge_ops(vocab, base_operations, num_merges) 
        morpho_segs = apply_merge_ops(preseg_vocab, morpho_operations, num_merges)
        morpho_segs = recover_preseg_boundary(vocab, presegs, morpho_segs)
        
        bpe_fmeasure =  call_evaluation(base_segs, eval_order, gold_standard_file, result_dir=bpe_folder)
        morph_fmeasure = call_evaluation(morpho_segs, eval_order, gold_standard_file, result_dir=morph_folder)

        num_symbols.append(num_merges)
        bpe_scores.append(bpe_fmeasure)
        morph_scores.append(morph_fmeasure)
        
        num_merges += granularity

    
    plt.plot(num_symbols, bpe_scores, "r")
    plt.plot(num_symbols, morph_scores, "b")
    plt.yticks(np.arange(20, 70, 5.0))
    plt.xlabel("BPE Iterations")
    plt.ylabel("F-Measure")
    plot_filename = os.path.join(args.output, "summary_plt.png")
    plt.savefig(plot_filename)

    #Write out a CSV of results in table form....
    results_csv = os.path.join(args.output, "results_table.csv")
    with open(results_csv, 'w+') as csvfile:
        results_writer = csv.writer(csvfile)
        results_writer.writerow(["Num Iterations", "BPE Scores", "Morpho Scores"])
        results_writer.writerow(["", "", ""])
        scores = zip(bpe_scores, morph_scores)
        for iterations, score_pair in zip(num_symbols, scores):
            results_writer.writerow([str(iterations) , str(score_pair[0]) , str(score_pair[1])])
        results_writer.writerow(["", "", ""])
        results_writer.writerow(["Morpho Peak: ", str(max(morph_scores))])
        results_writer.writerow(["BPE Peak: ", str(max(bpe_scores))])


    
    







    

    


