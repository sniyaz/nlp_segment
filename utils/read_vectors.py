import sys
import codecs
import argparse
import string
import numpy as np
import pickle
import pdb

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
    parser.add_argument(
        '--dim', '-d', action="store",
        metavar='PATH',
        help="Vector Dimension")
    
    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()
    dim = int(args.dim)

    word_vectors = {}

    for line in args.input:
        try:
            contents = [i for i in line.split()]
            vector = contents[-dim:]
            vector = [0.0 if i == "." else float(i) for i in vector]
            vector = np.asarray(vector)

            contents = contents[:-dim]
            word = "".join(contents)
            word_vectors[word] = vector
        except Exception as ec:
            pdb.set_trace()


    pickle.dump(word_vectors, args.output)








