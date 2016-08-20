#A script that scores segmentations bases on a gold standard.

import sys
import os
import codecs
import argparse
import string
sys.path.append("../")
from bpe import core_word_update
from collections import defaultdict, Counter

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
        '--merge_ops', '-ops', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Ordered list of merge operations that were taken by algo")
    parser.add_argument(
        '--symbols', '-s', action="store",
        help="Number of merge operations to perform.")

    return parser


def get_gs_data(input_obj, gold_standard, eval_order):
    for line in input_obj:
        line = line.strip()
        line = str(line)
        line_contents = line.split("\t")
        word = line_contents[0]
        word_segs = line_contents[1].split(", ")
        word_segs = [seg.split(" ") for seg in word_segs]
        gold_standard[word] = word_segs
        eval_order.append(word)




def apply_merge_ops(gold_standard, merge_ops_obj, num_symbols):
    merge_operations = []
    for line in merge_ops_obj:
        line = line.strip()
        line = str(line)
        pair = tuple(line.split(" "))
        merge_operations.append(pair)
    
    segmentations = {}
    quick_pairs = defaultdict(lambda: set())
    
    for word in gold_standard:
        #Set up segmentations data structure
        seg = list(word)
        #seg.append("</w>")
        segmentations[word] = seg
        #Set up the quick_find data structure
        for idx, c in enumerate(seg):
            #Now set up the quick_pairs data structure
            if idx != len(seg) - 1:
                quick_pairs[(c, seg[idx+1])].add((word, idx, idx+1))
    
    #Only do the first n merge operations...
    merge_operations = merge_operations[:int(num_symbols)]
    
    for pair in merge_operations:
        new_symbol = "".join(pair)
        #Some of the pairs aren't relevant to the evaluations set...
        if pair in quick_pairs:
            involved_words = quick_pairs[pair]

            while involved_words:
                word, first_index, second_index = involved_words.pop()
                #Call this with throw away dicts for the frequencey cache. Not relevant here at all.
                core_word_update(word, pair, new_symbol, first_index, second_index, quick_pairs, segmentations, Counter(), False)
            
            quick_pairs.pop(pair)

    return segmentations


def extract_fmeasure(result_filename):
    result_obj = open(result_filename, "r")
    for line in result_obj:
        line = line.strip()
        line = line.split("  ")
        if line[0] == "F-measure:":
            return float(line[1][:-1])
    result_obj.close()


def call_evaluation(segmentations, eval_order, gold_standard_path, result_dir=None):
    if result_dir is None:
        os.system("mkdir eval_temp")
        segs_output_obj = open("eval_temp/segs.txt", "w+")
    else:
        segs_output_obj = open(os.path.join(result_dir, "segs.txt"), "w+")
    for word in eval_order:
        final_seg = segmentations[word]
        delimited_seg = " ".join(final_seg)
        delimited_seg = delimited_seg.replace("</w>", "")
        segs_output_obj.write(delimited_seg + '\n')
    segs_output_obj.close()

    if result_dir is None:
        os.system("perl evaluation.perl -desired " + gold_standard_path + \
            " -suggested eval_temp/segs.txt > eval_temp/eval_output.txt")
        
        os.system("rm -r eval_temp")
    else:
        os.system("perl evaluation.perl -desired " + gold_standard_path + " -suggested " + str(os.path.join(result_dir, "segs.txt")) \
            + " > " + os.path.join(result_dir, "eval_output.txt"))
        fmeasure = extract_fmeasure(os.path.join(result_dir, "eval_output.txt"))

    return fmeasure


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

    gold_standard = {}
    eval_order = []
    get_gs_data(args.input, gold_standard, eval_order)
    
    segmentations = apply_merge_ops(gold_standard, args.merge_ops, args.symbols)

    call_evaluation(segmentations, eval_order, args.input.name)






    

    




