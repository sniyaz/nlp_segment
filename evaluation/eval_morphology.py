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

from evaluate_seg import get_gs_data, call_evaluation, apply_merge_ops, extract_fmeasure
sys.path.append("../")
from morphology_preprocess import compute_preseg, process_json
from bpe import get_vocabulary_freq_table, apply_presegs, recover_preseg_boundary

import pdb

def eval_mofessor(model_path, gs_path, gs_wordlist_path, result_dir):

    segs_file = os.path.join(result_dir, "segs.txt")
    os.system("morfessor-segment -L " + model_path + " -o" + segs_file + \
         " " + gs_wordlist_path)
    
    #Now evaluate those segmentations' score
    eval_result_file = os.path.join(result_dir, "eval_output.txt")
    os.system("perl evaluation.perl -desired " + gs_path + " -suggested " + segs_file \
            + " > " + eval_result_file)
    return extract_fmeasure(eval_result_file)
    

# hack for python2/3 compatibility
from io import open
argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', action="store",
        metavar='PATH',
        help="Directory that has the data to use in evalution")
    parser.add_argument(
        '--morfessor_model', '-m', action="store",
        metavar='PATH',
        help="Model to the morfessor model to use for comparison")
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

    test_corpus_file = os.path.join(args.input, "pure_corpus.txt")
    gold_standard_file = os.path.join(args.input, "gs_corpus_only.txt")
    gold_standard_words = os.path.join(args.input, "gs_clean_wordlist.txt")

    gold_standard = {}
    eval_order = []
    get_gs_data(open(gold_standard_file, "r"), gold_standard, eval_order)

    presegs = pickle.load(args.pre_segs)
 
    word_vectors = pickle.load(open("/Users/Sherdil/Research/NLP/nlp_segment/data/vectors.txt", "rb"))
    vocab = get_vocabulary_freq_table(open(test_corpus_file, "r"), word_vectors)  
    segmented_vocab = apply_presegs(copy.deepcopy(vocab), presegs)
    test = json.load(open("../data/morph_rules.json", "r"))

    #Run morfessor as a baseline to compare against.
    morfessor_results = os.path.join(args.output, "morfessor_results")
    os.system("mkdir " + morfessor_results)
    morfessor_fmeasure = eval_mofessor(args.morfessor_model, gold_standard_file, gold_standard_words, morfessor_results)
    
    os.system("python3 ../bpe.py --mode 3 -i " + test_corpus_file + " -ft " + \
    " -o " + os.path.join(args.output, "preseg") +  " -s " + str(max_merges))

    os.system("python3 ../bpe.py --mode 1 -i " + test_corpus_file + " -ft " + \
    " -o " + os.path.join(args.output, "base") +  " -s " + str(max_merges))
    

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

        base_merge_ops_obj = open(os.path.join(args.output, "base_merge_ops.txt"))
        gs_bpe_segs = apply_merge_ops(gold_standard, base_merge_ops_obj, num_merges) 
        base_merge_ops_obj.close()

        preseg_merge_ops_obj = open(os.path.join(args.output, "preseg_merge_ops.txt"))
        preseg_segs = apply_merge_ops(segmented_vocab, preseg_merge_ops_obj, num_merges) 
        gs_mixed_segs = recover_preseg_boundary(gold_standard, presegs, preseg_segs)
        preseg_merge_ops_obj.close()

        # gs_mixed_segs = {}
        # for word in gold_standard:
        #     if len(presegs[word]) == 1:
        #         gs_mixed_segs[word] = gs_bpe_segs[word]
        #     else:
        #         gs_mixed_segs[word] = presegs[word]
        
        bpe_fmeasure =  call_evaluation(gs_bpe_segs, eval_order, gold_standard_file, result_dir=bpe_folder)
        morph_fmeasure = call_evaluation(gs_mixed_segs, eval_order, gold_standard_file, result_dir=morph_folder)

        num_symbols.append(num_merges)
        bpe_scores.append(bpe_fmeasure)
        morph_scores.append(morph_fmeasure)
        
        num_merges += granularity

    
    plt.plot(num_symbols, bpe_scores, "r")
    plt.plot(num_symbols, morph_scores, "b")
    #Add morfessor baseline to plot
    plt.axhline(y=morfessor_fmeasure, linewidth=2, color='g')
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
        results_writer.writerow(["Morfessor Baseline: ", str(morfessor_fmeasure)])
        results_writer.writerow(["Morpho Peak: ", str(max(morph_scores))])
        results_writer.writerow(["BPE Peak: ", str(max(bpe_scores))])


    
    







    

    


