#A script that scores segmentations bases on a gold standard.

import sys
import os
import codecs
import argparse
import string
sys.path.append("../")
from segment import core_word_update
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

    return parser


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

    gold_standard = {}
    eval_order = []
    for line in args.input:
        line = line.strip()
        line = str(line)
        line_contents = line.split("\t")
        word = line_contents[0]
        word_segs = line_contents[1].split(", ")
        word_segs = [seg.split(" ") for seg in word_segs]
        gold_standard[word] = word_segs
        eval_order.append(word)
    
    merge_operations = []
    for line in args.merge_ops:
        line = line.strip()
        line = str(line)
        pair = tuple(line.split(" "))
        merge_operations.append(pair)


    segmentations = {}
    quick_pairs = {}
    for word in gold_standard:
        #Set up segmentations data structure
        segmentations[word] = list(word)
        #Set up the quick_pairs data structure
        for idx, c in enumerate(word):
            if idx != len(word) - 1:
                if (c, word[idx+1]) not in quick_pairs:
                    quick_pairs[(c, word[idx+1])] = set([(word, idx, idx+1)])
                else:
                    quick_pairs[(c, word[idx+1])].add((word, idx, idx+1))
    

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

    
    os.system("mkdir eval_temp")
    
    segs_output_obj = open("eval_temp/segs.txt", "w+")
    for word in eval_order:
        final_seg = segmentations[word]
        delimited_seg = " ".join(final_seg)
        segs_output_obj.write(delimited_seg + '\n')
    segs_output_obj.close()

    os.system("perl evaluation.perl -desired " + args.input.name + " -suggested eval_temp/segs.txt")
    os.system("rm -r eval_temp")






    

    



