import sys
import os
import codecs
import argparse
import string
import pickle
import json
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

from evaluate_seg import get_gs_data, call_evaluation, apply_merge_ops
from run_experiments import extract_fmeasure
sys.path.append("../")
from morphology_preprocess import compute_preseg, process_json
from bpe import get_vocabulary_freq_table

import pdb


# hack for python2/3 compatibility
from io import open
argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="File of goldstandard segmentations")
    parser.add_argument(
        '--corpus', '-c', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Corpus for evaluation")
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

    gold_standard = {}
    eval_order = []
    get_gs_data(args.input, gold_standard, eval_order)

    word_vectors = pickle.load(open("/Users/Sherdil/Research/NLP/nlp_segment/data/vectors.txt", "rb"))
    vocab = get_vocabulary_freq_table(args.corpus, word_vectors)  
    test = json.load(open("../data/morph_rules.json", "r"))
    prefix_transforms, suffix_transforms = process_json(test) 
    presegs = pickle.load(open("../debug_temp/presegs_ckpt.txt", "rb"))

    os.system("python3 ../bpe.py --mode 1 -i " + args.corpus.name + " -ft " + \
    " -o " + os.path.join(args.output, "exp") +  " -s " + str(max_merges))
    

    num_symbols = []
    bpe_scores = []
    morph_scores = []
    
    num_merges = granularity
    while (num_merges <= max_merges):
        save_folder = os.path.join(args.output, str(num_merges) + "_merges")
        os.system("mkdir " + save_folder)
        bpe_folder = os.path.join(save_folder, "bpe_folder")
        os.system("mkdir " + bpe_folder)
        morph_folder = os.path.join(save_folder, "morph_folder")
        os.system("mkdir " + morph_folder)
        bpe_eval_file = os.path.join(bpe_folder, "eval_output.txt")
        morph_eval_file = os.path.join(morph_folder, "eval_output.txt")

        merge_ops_obj = open(os.path.join(args.output, "exp_merge_ops.txt"))
        bpe_segs = apply_merge_ops(gold_standard, merge_ops_obj, num_merges)
        merge_ops_obj.close()
        #presegs = compute_preseg(vocab, word_vectors, prefix_transforms, suffix_transforms, test_set=list(gold_standard.keys()))
        mixed_segs = {}
        for word in gold_standard:
            if len(presegs[word]) == 1:
                mixed_segs[word] = bpe_segs[word]
            else:
                mixed_segs[word] = presegs[word]
        
        #with open(os.path.join(args.output, "presegs_ckpt.txt"), "wb+") as checkpoint_file:
            #pickle.dump(presegs, checkpoint_file)

        call_evaluation(bpe_segs, eval_order, args.input.name, result_dir=bpe_folder)
        call_evaluation(mixed_segs, eval_order, args.input.name, result_dir=morph_folder)
            
        num_symbols.append(num_merges)
        bpe_scores.append(extract_fmeasure(bpe_eval_file))
        morph_scores.append(extract_fmeasure(morph_eval_file))
        
        num_merges += granularity

    plt.plot(num_symbols, bpe_scores, "r")
    plt.plot(num_symbols, bpe_scores, "ro")
    plt.plot(num_symbols, morph_scores, "b")
    plt.plot(num_symbols, morph_scores, "bo")
    plot_filename = os.path.join(args.output, "summary_plt.png")
    plt.savefig(plot_filename)
    
    







    

    


