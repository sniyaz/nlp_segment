#Combine 2005/2010 gold standards for any corpus.

from __future__ import unicode_literals, division

import sys
import codecs
import argparse
import string
import os
from collections import defaultdict, Counter

import numpy as np
import pickle

from prepare_seg_eval_data import remove_colons

sys.path.append("../")
from bpe import get_vocabulary

import pdb

# hack for python2/3 compatibility
from io import open
argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--corpus', '-c', type=argparse.FileType(mode='r', encoding='utf-8', errors='ignore'), default=sys.stdin,
        metavar='PATH',
        help="Input corpus")
    parser.add_argument(
        '--gold_standard_1', '-gs1', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="2005 gold standard to clean.")
    parser.add_argument(
        '--gold_standard_2', '-gs2', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="2010 gold standard to clean.")
    parser.add_argument(
        '--output', '-o', action="store",
        metavar='PATH',
        help="Output path")
    
    return parser

        

if __name__ == '__main__':
         
    parser = create_parser()
    args = parser.parse_args()

    vocab = get_vocabulary(args.corpus)
    clean_gold_standard = open(args.output, "w+")

    gold_words = []

    for line in args.gold_standard_1:
        line = line.strip()
        line = str(line)
        line_contents = line.split("\t")
        word = line_contents[0]
        if word in vocab:
            clean_gold_standard.write(line)
            clean_gold_standard.write("\n")
            gold_words.append(word)

    for line in args.gold_standard_2:
        line = line.strip()
        line = str(line)
        line_contents = line.split("\t")
        word = line_contents[0]
        if word in vocab and (word not in gold_words):
            word_segs = line_contents[1].split(", ")
            word_segs = [seg.split(" ") for seg in word_segs]
            word_segs = [remove_colons(seg) for seg in word_segs]
            word_segs = [" ".join(seg) for seg in word_segs]
            new_line = ", ".join(word_segs)
            new_line = word + "\t" + new_line + "\n"
            clean_gold_standard.write(new_line)
            

            

    clean_gold_standard.close()
            


    



