from __future__ import unicode_literals, division

import sys
import codecs
import argparse
import string
import os
from collections import defaultdict, Counter

import numpy as np
import pickle

sys.path.append("../")
from bpe import get_vocabulary_freq_table

import pdb

# hack for python2/3 compatibility
from io import open
argparse.open = open


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Clean training corpus and goldstandard for evaluation..")
    parser.add_argument(
        '--corpus', '-c', type=argparse.FileType(mode='r', encoding='utf-8', errors='ignore'), default=sys.stdin,
        metavar='PATH',
        help="Input corpus to clean.")
    parser.add_argument(
        '--gold_standard', '-gs', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input gold standard to clean.")
    parser.add_argument(
        '--output', '-o', action="store",
        metavar='PATH',
        help="Output dir")
    
    return parser


def remove_colons(seg):
    new_seg = []
    for part in seg:
        if ":" in part:
            colon_index = part.index(":")
            part = part[:colon_index]
        if part != "~":
            new_seg.append(part)

    return new_seg
        

if __name__ == '__main__':
         
    parser = create_parser()
    args = parser.parse_args()

    word_vectors = pickle.load(open("/Users/Sherdil/Research/NLP/nlp_segment/data/vectors.txt", "rb"))

    vocab = get_vocabulary_freq_table(args.corpus, word_vectors)
    vocab = [word for word in vocab if word in word_vectors]

    clean_corpus = open(os.path.join(args.output, "pure_corpus.txt"), "w+")
    clean_gold_standard = open(os.path.join(args.output, "gs_corpus_only.txt"), "w+")
    clean_gold_standard_wordlist = open(os.path.join(args.output, "gs_clean_wordlist.txt"), "w+")

    args.corpus.seek(0)
    for line in args.corpus:
        original_line = line
        line = line.strip()
        line_parts = line.split()
        if len(line_parts) != 2:
            continue
        word = line_parts[1]
        if word in vocab:
            clean_corpus.write(original_line)
    

    for line in args.gold_standard:
        line = line.strip()
        line = str(line)
        line_contents = line.split("\t")
        word = line_contents[0]
        if word in vocab:
            clean_gold_standard_wordlist.write(word+ "\n")
            word_segs = line_contents[1].split(", ")
            word_segs = [seg.split(" ") for seg in word_segs]
            word_segs = [remove_colons(seg) for seg in word_segs]
            word_segs = [" ".join(seg) for seg in word_segs]
            new_line = ", ".join(word_segs)
            new_line = word + "\t" + new_line + "\n"
            clean_gold_standard.write(new_line)
   
    clean_corpus.close()
    clean_gold_standard.close()
    clean_gold_standard_wordlist.close()
            


    



