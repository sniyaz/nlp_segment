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
from collections import defaultdict, Counter

from evaluate_seg import get_gs_data, call_evaluation, extract_fmeasure, get_merge_ops_list
sys.path.append("../")
from bpe import get_vocabulary, apply_presegs, recover_preseg_boundary, apply_merge_ops, segment_vocab, extract_boundaries

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

#Thank you Papa DeNero for teaching me environment diagrams.
def base_valid_func(base_segs, op_number):
    num_merges = op_number
    save_folder = os.path.join(args.output, str(num_merges) + "_merges")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    bpe_folder = os.path.join(save_folder, "bpe_folder")
    os.makedirs(bpe_folder)
    bpe_eval_file = os.path.join(bpe_folder, "eval_output.txt")
    bpe_fmeasure =  call_evaluation(base_segs, eval_order, gold_standard_file, result_dir=bpe_folder)
    return bpe_fmeasure

def morpho_valid_func(morpho_segs, op_number):
    num_merges = op_number 
    save_folder = os.path.join(args.output, str(num_merges) + "_merges")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    morph_folder = os.path.join(save_folder, "morph_folder")
    os.makedirs(morph_folder)
    morph_eval_file = os.path.join(morph_folder, "eval_output.txt")
    #TODO: TAKE OUUUUTTT
    #morpho_segs = recover_preseg_boundary(vocab, presegs, morpho_segs)
    morph_fmeasure = call_evaluation(morpho_segs, eval_order, gold_standard_file, result_dir=morph_folder)
    return morph_fmeasure

def gen_result_summary(num_symbols, bpe_scores, morph_scores, output_folder):
    plt.plot(num_symbols, bpe_scores, sns.xkcd_rgb["pale red"], lw=3)
    plt.plot(num_symbols, morph_scores, sns.xkcd_rgb["denim blue"], lw=3)
    plt.yticks(np.arange(20, 70, 5.0))
    plt.xlabel("Merge Operations")
    plt.ylabel("F-measure")
    blue_patch = mpatches.Patch(color=sns.xkcd_rgb["denim blue"], label='BPE with pre-segmentations')
    red_patch = mpatches.Patch(color=sns.xkcd_rgb["pale red"], label='BPE')
    plt.legend(handles=[blue_patch, red_patch])
    plot_filename = os.path.join(output_folder, "summary_plt.png")
    plt.savefig(plot_filename)

    #Write out a CSV of results in table form....
    results_csv = os.path.join(output_folder, "results_table.csv")
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
    # preseg_vocab = apply_presegs(copy.deepcopy(vocab), presegs)

    # _, __, bpe_scores = segment_vocab(vocab, max_merges, valid_freq=granularity, valid_func=base_valid_func)
    # _, __, morph_scores= segment_vocab(preseg_vocab, max_merges, valid_freq=granularity, valid_func=morpho_valid_func)

    #BOUNDARY BASED APPROACH!
    boundaries = extract_boundaries(vocab, presegs)
    _, __, bpe_scores = segment_vocab(copy.deepcopy(vocab), max_merges, valid_freq=granularity, valid_func=base_valid_func)
    _, __, morph_scores= segment_vocab(copy.deepcopy(vocab), max_merges, valid_freq=granularity, valid_func=morpho_valid_func, boundaries=boundaries)

    num_symbols = list(range(0, max_merges + 1, granularity))

    gen_result_summary(num_symbols, bpe_scores, morph_scores, args.output)

    
    







    

    


