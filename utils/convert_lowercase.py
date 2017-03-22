import sys

"""
Convert corpus to lowercase. Needed for training vectors for pre-segs that ignore case.
"""

corpus_path = sys.argv[1]
output_path = sys.argv[2]

corpus_obj = open(corpus_path, "r")
output_obj = open(output_path, "w+")
for line in corpus_obj:
    new_line = []
    for word in line.split():
        lower_word = word.lower()
        new_line.append(lower_word)
    new_line = " ".join(new_line)
    output_obj.write(new_line)
    output_obj.write('\n')

corpus_obj.close()
output_obj.close()