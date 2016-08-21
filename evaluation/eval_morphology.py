import sys
import os
import codecs
import argparse
import string
import pickle
import json
import copy
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

from evaluate_seg import get_gs_data, call_evaluation, apply_merge_ops
sys.path.append("../")
from morphology_preprocess import compute_preseg, process_json
from bpe import get_vocabulary_freq_table, apply_presegs, recover_preseg_boundary

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

    with open("../debug_temp/presegs_ckpt.txt", "rb") as checkpoint_file:
        presegs = pickle.load(checkpoint_file)
    #presegs = compute_preseg(vocab, word_vectors, morph_transforms, test_set=list(gold_standard.keys()))
    #with open(os.path.join(args.output, "presegs_ckpt.txt"), "wb+") as checkpoint_file:
        #pickle.dump(presegs, checkpoint_file)

    word_vectors = pickle.load(open("/Users/Sherdil/Research/NLP/nlp_segment/data/vectors.txt", "rb"))
    vocab = get_vocabulary_freq_table(args.corpus, word_vectors)  
    segmented_vocab = apply_presegs(copy.deepcopy(vocab), presegs)
    test = json.load(open("../data/morph_rules.json", "r"))
    morph_transforms = process_json(test) 

    
    os.system("python3 ../bpe.py --mode 3 -i " + args.corpus.name + " -ft " + \
    " -o " + os.path.join(args.output, "preseg") +  " -s " + str(max_merges))

    os.system("python3 ../bpe.py --mode 1 -i " + args.corpus.name + " -ft " + \
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
        
        bpe_fmeasure =  call_evaluation(gs_bpe_segs, eval_order, args.input.name, result_dir=bpe_folder)
        morph_fmeasure = call_evaluation(gs_mixed_segs, eval_order, args.input.name, result_dir=morph_folder)
            
        num_symbols.append(num_merges)
        bpe_scores.append(bpe_fmeasure)
        morph_scores.append(morph_fmeasure)
        
        num_merges += granularity

    plt.plot(num_symbols, bpe_scores, "r")
    #plt.plot(num_symbols, bpe_scores, "ro")
    plt.plot(num_symbols, morph_scores, "b")
    #plt.plot(num_symbols, morph_scores, "bo")
    plot_filename = os.path.join(args.output, "summary_plt.png")
    plt.savefig(plot_filename)
    
    







    

    


