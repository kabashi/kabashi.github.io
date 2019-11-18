#!/usr/bin/env python3

import argparse
import re


def arguments():
    parser = argparse.ArgumentParser(description="Split a BibTeX file into individual entries.")
    parser.add_argument("FILE", type=argparse.FileType("r"), help="The input file")
    args = parser.parse_args()
    return args


def entry_generator(f):
    find_key = re.compile(r'^@[^{]+\{(.+),$')
    in_entry = False
    entry = None
    key = None
    for line in f:
        line = line.rstrip()
        if line == "":
            continue
        elif line.startswith("@") and line.endswith(","):
            m = find_key.search(line)
            if m:
                key = m.group(1)
            in_entry = True
            entry = [line]
        elif line == "}":
            in_entry = False
            entry.append(line)
            yield key, entry
            entry = None
            key = None
        elif in_entry:
            entry.append(line)


def main():
    args = arguments()
    for key, entry in entry_generator(args.FILE):
        with open("../bib/%s.bib" % key, mode="w") as out:
            out.write("\n".join(entry))
            out.write("\n")


if __name__ == "__main__":
    main()
