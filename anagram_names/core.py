"""Library"""

__author__ = "John Evans"
__copyright__ = "Copyright 2023, John Evans"
__credits__ = ["John Evans"]
__license__ = "GPLv3.0"
__version__ = "0.1.0"
__maintainer__ = "John Evans"
__email__ = "thejevans@pm.me"
__status__ = "Production"


import collections
import functools
import itertools
import json

import nltk
from nltk import FreqDist
from nltk.corpus import brown, words
from numba import njit, prange
import numpy as np
from tqdm.notebook import tqdm


def logging(log):
    def logging_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            print(f'{log}...', end='', flush=True)
            result = func(*args, **kwargs)
            print('done.')
            return result
        return wrapper
    return logging_decorator


@logging('finding all anagram sets in list')
@njit
def word_to_word_anagrams(word_arr: np.ndarray) -> np.ndarray:
    """
    uses bit packing to keep memory usage reasonable.
    
    this finds all words that are anagrams of each other
    """
    bitmap = np.array([
        np.uint8(128),
        np.uint8(64),
        np.uint8(32),
        np.uint8(16),
        np.uint8(8),
        np.uint8(4),
        np.uint8(2),
        np.uint8(1),
    ], dtype=np.uint8)

    result_arr = np.zeros((len(word_arr), int(np.ceil(len(word_arr) / 8))), dtype=np.uint8)
    
    # without this, it's 10 times slower. no idea why.
    dummy = np.zeros(26, dtype=np.int8)

    for i in range(len(word_arr)):
        result_arr[i] = _word_to_word_row(bitmap, i, word_arr, dummy)

    return result_arr


@njit(parallel=True)
def _word_to_word_row(bitmap: np.ndarray, i: int, word_arr: np.ndarray, dummy: np.ndarray) -> np.ndarray:
    row_arr = np.zeros(int(np.ceil(len(word_arr) / 8)), dtype=np.uint8)

    for j in prange(len(word_arr)):
        counts = word_arr[i] - word_arr[j] + dummy

        if not np.any(counts):
            row_arr[j // 8] += bitmap[j % 8]

    return row_arr


@logging('filtering word pairs for potential anagrams')
@njit
def last_name_filter(word_arr: np.ndarray, snarks_arr: np.ndarray) -> np.ndarray:
    """
    uses bit packing to keep memory usage reasonable.
    
    this reduces the search space to only pairs of words with snarks in them somewhere.
    """
    bitmap = np.array([
        np.uint8(128),
        np.uint8(64),
        np.uint8(32),
        np.uint8(16),
        np.uint8(8),
        np.uint8(4),
        np.uint8(2),
        np.uint8(1),
    ], dtype=np.uint8)

    result_arr = np.zeros((len(word_arr), int(np.ceil(len(word_arr) / 8))), dtype=np.uint8)

    for i in range(len(word_arr)):
        result_arr[i] = _snarks_row(bitmap, i, word_arr, snarks_arr)

    return result_arr


@njit(parallel=True)
def _snarks_row(
        bitmap: np.ndarray, i: int, word_arr: np.ndarray, snarks_arr: np.ndarray) -> np.ndarray:
    row_arr = np.zeros(int(np.ceil(len(word_arr) / 8)), dtype=np.uint8)

    for j in prange(i + 1, len(word_arr)):
        counts = word_arr[i] + word_arr[j] - snarks_arr

        if not np.any(counts < 0) and np.sum(counts) > 0:
            row_arr[j // 8] += bitmap[j % 8]

    return row_arr


@njit(parallel=True)
def snarks_anagram_row(
    i: int,
    row: np.ndarray,
    snarks_count: np.ndarray,
    word_arr: np.ndarray,
    word_hash_idxs: np.ndarray,
    bitmap: np.ndarray,
) -> np.ndarray:
    """
    this does most of the heavy lifting of actually finding valid anagrams

    uses the fact that words are sorted by hash to search fewer words
    """
    potential_match_idxs = np.nonzero(row)[0]
    row_matches = -1 * np.ones(len(potential_match_idxs), dtype=np.int32)
    remainder_counts = word_arr[i] + word_arr[potential_match_idxs] - snarks_count

    for j in prange(len(potential_match_idxs)):
        remainder_hash = 0
        for k in range(8):
            if remainder_counts[j, k] == 0:
                remainder_hash += bitmap[k]
            if remainder_counts[j, k + 8] == 0:
                remainder_hash += bitmap[k]
            if remainder_counts[j, k + 16] == 0:
                remainder_hash += bitmap[k]
        for k in range(2):
            if remainder_counts[j, k + 24] == 0:
                remainder_hash += bitmap[k]

        lo = word_hash_idxs[remainder_hash][0]
        hi = word_hash_idxs[remainder_hash][1]
        abs_diff_sums = np.sum(np.abs(word_arr[lo:hi] - remainder_counts[j]), axis=1)
        matches = np.nonzero(abs_diff_sums == 0)[0]
        if len(matches) > 0:
            # since all one-word anagrams are gone, there should only be one match
            row_matches[j] = matches[0] + lo

    valid = np.nonzero(row_matches != -1)[0]  
    return potential_match_idxs[valid], row_matches[valid]


@logging('loading word list')
def load_wordlist(corpus: str) -> list[str]:
    nltk.download(corpus)

    if corpus == 'words':
        from nltk.corpus import words
    elif corpus == 'brown':
        from nltk.corpus import brown

    word_list = [s.lower() for s in set(corpus.words())]
    return list(set(word_list))


@logging('filtering word list')
def filter_wordlist(word_list: list[str]) -> list[str]:
    word_list = [word.replace('-', '') for word in word_list]
    word_list = [word.replace('.', '') for word in word_list]
    return [word for word in word_list if word.isalpha()]


@logging('building word count arrays')
def build_count_arrs(word_list: list[str], snarks: str) -> tuple[np.ndarray, np.ndarray]:
    word_count_arr = np.zeros((len(word_list), 26), dtype=np.int8)
    snarks_count_arr = np.zeros(26, dtype=np.int8)

    for i, word in enumerate(word_list):
        count = collections.Counter(word)
        for char, num in count.items():
                word_count_arr[i, ord(char) - ord('a')] = num

    for char, num in collections.Counter(snarks).items():
        snarks_count_arr[ord(char.lower()) - ord('a')] = num

    return word_count_arr, snarks_count_arr


@logging('sorting word list by hash')
def sort_wordlist(word_list: list[str], word_count_arr: np.ndarray) -> list[str]:
    hash_arr = hash_func(word_count_arr)
    return list(zip(*sorted(zip(hash_arr, word_list))))[1]


def hash_func(word_count_arr: np.ndarray):
    return np.sum(
        np.packbits(word_count_arr == 0, bitorder='little', axis=-1),
        axis=-1,
        dtype=np.uint16,
    )


@logging('condensing word list')
def condense_words_by_anagrams(word_list: list[str], anagram_arr: np.ndarray):
    anagram_map = {word: [] for word in word_list}
    for i, word in enumerate(word_list):
        row = np.unpackbits(anagram_arr[i])
        anagram_idxs = np.nonzero(row)[0]
        anagram_map[word] = [word_list[j] for j in anagram_idxs if word_list[j] != word]

    condensed_word_list = []
    for word, anagrams in anagram_map.items():
        for anagram in anagrams:
            if anagram in condensed_word_list:
                break
        else:
            condensed_word_list.append(word)
            
    anagram_map = {word: anagram_map[word] for word in condensed_word_list}
    
    return condensed_word_list, anagram_map


@logging('building word hash index array')
def hash_idx_arr(word_list: list[str], word_count_arr: np.ndarray) -> np.ndarray:
    hashes = hash_func(word_count_arr)
    word_hash_idxs = np.zeros((np.amax(hashes) + 1, 2), dtype=np.uint32)

    for j, hash_val in enumerate(hashes):
        if word_hash_idxs[hash_val, 0] == 0:
            word_hash_idxs[hash_val, 0] = j
        if hash_val > 0:
            word_hash_idxs[hashes[j - 1], 1] = j

    word_hash_idxs[np.amax(hashes), 1] = len(word_list)
    return word_hash_idxs


@logging('finding all valid anagrams')
def find_anagrams(
    word_list: list[str],
    word_hash_idxs: np.ndarray,
    word_count_arr: np.ndarray,
    snarks_count_arr: np.ndarray,
    filter_arr: np.ndarray,
) -> dict[str, list[tuple[str, str]]]:
    bitmap = 2**np.array(np.arange(8), dtype=np.uint8)

    pairs = {}
    for i in tqdm(range(len(word_list))):
        row = np.unpackbits(filter_arr[i])
        pairing_idxs, name_idxs = snarks_anagram_row(
            i, row, snarks_count_arr, word_count_arr, word_hash_idxs, bitmap)

        for name_idx, pairing_idx in zip(name_idxs, pairing_idxs):
            name = word_list[name_idx]
            a = word_list[pairing_idx]
            b = word_list[i]
            if name not in pairs:
                pairs[name] = []
            pairs[name].append((min(a, b), max(a, b)))
        
    return pairs


@logging('building anagram dictionary')
def build_anagram_dict(
    snarks: str,
    pairs: dict[str, list[tuple[str, str]]],
    anagram_map: dict[str, list[str]],
) -> dict[str, dict[str, list[str]] | list[str]]:
    anagrams = {}
    for word, pair_list in pairs.items():
        full_pair_list = [*pair_list]
        for a, b in pair_list:
            full_pair_list.extend(
                list(itertools.product(anagram_map[a], anagram_map[b])))
        pair_strs = [f'{i} {j}' for i, j in full_pair_list]
        if len(anagram_map[word]) > 0:
            anagrams[f'{word} {snarks}'] = {
                'whole name anagrams': pair_strs,
                'first name anagrams': anagram_map[word],
            }
        else:
            anagrams[f'{word} {snarks}'] = pair_strs
    return anagrams


@logging('saving anagrams to file(s)')
def save_anagrams(
    anagrams: dict[str, list[str] | dict[str, list[str]]],
    out_dir: str,
) -> None:
    first_name_lens = [len(name[:name.find(' ')]) for name in anagrams]
    longest_first_name = max(first_name_lens)
    test_name = next(iter(anagrams))
    snarks_length = len(test_name[test_name.find(' '):])
    out_dicts = [{} for _ in range(longest_first_name)]
    for name, name_anagrams in anagrams.items():
        out_dicts[len(name) - snarks_length - 1][name] = name_anagrams
    for i, out_dict in enumerate(out_dicts):
        out_file = f'{out_dir}/{i + 1:02} char names.json'
        with open(out_file, 'w') as f:
            json.dump(out_dict, f)
