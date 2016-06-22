import sys
import codecs
import argparse
import string
import numpy as np
import pickle


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="read in vector representations of words")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('wb'), default=sys.stdout,
        metavar='PATH',
        help="Output file for the dictionary of word vectors (default: standard output)")

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    word_vectors = {}

    for line in args.input:
        contents = [i for i in line.split()]
        word = contents[0]
        contents = contents[1:]
        contents = [float(i) for i in contents]
        vector = np.asarray(contents)
        word_vectors[word] = vector

    pickle.dump(word_vectors, args.output)








