"""Main program"""

__author__ = "John Evans"
__copyright__ = "Copyright 2023, John Evans"
__credits__ = ["John Evans"]
__license__ = "GPLv3.0"
__version__ = "0.1.0"
__maintainer__ = "John Evans"
__email__ = "thejevans@pm.me"
__status__ = "Production"


import argparse

from . import core


CORPORA = ['words', 'brown']


def main() -> None:
    """Docstring"""
    args = parse_args()
    uncondensed_word_list = load_words(args['corpus'])
    print(f'length of uncondensed word list: {len(uncondensed_word_list)}')
    word_list, anagram_map = condense_words(uncondensed_word_list, args['last_name'])
    del uncondensed_word_list
    print(f'length of condensed word list: {len(word_list)}')
    anagrams = find_anagrams(word_list, anagram_map, args['last_name'])
    core.save_anagrams(anagrams, args['out_dir'])


def parse_args() -> argparse.Namespace:
    """Docstring"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--corpus', choices=CORPORA, default=CORPORA[0])
    #parser.add_argument('-f', '--first-names', default=CORPORA[0])
    parser.add_argument('last_name')
    parser.add_argument('out_dir')
    return parser.parse_args()


def load_words(corpus: str) -> list[str]:
    """Docstring"""
    uncondensed_word_list = core.load_wordlist(corpus)
    return core.filter_wordlist(uncondensed_word_list)


def condense_words(uncondensed_word_list: list[str], last_name: str) -> tuple[list[str], dict]:
    """Docstring"""
    uncondensed_word_count_arr, _ = core.build_count_arrs(uncondensed_word_list, last_name)
    uncondensed_word_list = core.sort_wordlist(uncondensed_word_list, uncondensed_word_count_arr)
    uncondensed_word_count_arr, _ = core.build_count_arrs(uncondensed_word_list, last_name)
    anagram_arr = core.word_to_word_anagrams(uncondensed_word_count_arr)
    word_list, anagram_map = core.condense_words_by_anagrams(uncondensed_word_list, anagram_arr)

    del anagram_arr
    del uncondensed_word_count_arr

    return word_list, anagram_map


def find_anagrams(word_list: list[str], anagram_map: dict, last_name: str) -> dict:
    """Docstring"""
    word_count_arr, last_name_count_arr = core.build_count_arrs(word_list, last_name)
    word_hash_idxs = core.hash_idx_arr(word_list, word_count_arr)
    filter_arr = core.last_name_filter(word_count_arr, last_name_count_arr)
    pairs = core.find_anagrams(word_list, word_hash_idxs, word_count_arr, last_name_count_arr, filter_arr)
    return core.build_anagram_dict(last_name, pairs, anagram_map)


if __name__ == "__main__":
    main()
