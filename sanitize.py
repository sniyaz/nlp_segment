from __future__ import unicode_literals, division

import sys
import codecs
import argparse
import string
from collections import defaultdict


# hack for python2/3 compatibility
from io import open
argparse.open = open

# python 2/3 compatibility
if sys.version_info < (3, 0):
  sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
  sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
  sys.stdin = codecs.getreader('UTF-8')(sys.stdin)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('a+'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")

    return parser


if __name__ == '__main__':
    parser = create_parser()
    args = parser.parse_args()

    for line in args.input:
        contents = line.split()
        new_contents = []
        for word in contents:
           new_word = word.encode('utf-8').translate(None, string.punctuation)
           new_word = new_word.lower()
           new_word = new_word.decode('utf-8')
           new_contents.append(new_word)
        new_line = " ".join(new_contents)
        args.output.write(new_line)
        args.output.write('\n')

