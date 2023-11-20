import datetime
import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import interface


def test_levenshtein_distance_same_strings():
    assert interface.LevenshteinDistance.calculate_distance("abc", "abc") == 0


def test_levenshtein_distance_one_empty_string():
    assert interface.LevenshteinDistance.calculate_distance("abc", "") == 3
    assert interface.LevenshteinDistance.calculate_distance("", "xyz") == 3


def test_levenshtein_distance_strings_with_same_length():
    assert interface.LevenshteinDistance.calculate_distance("kitten", "sitting") == 3


def test_levenshtein_distance_strings_with_different_length():
    assert interface.LevenshteinDistance.calculate_distance("kitten", "kit") == 3
    assert interface.LevenshteinDistance.calculate_distance("kit", "kitten") == 3

def test_levenshtein_distance_large_strings():
    s1 = "a" * 1000
    s2 = "b" * 1000
    assert interface.LevenshteinDistance.calculate_distance(s1, s2) == 1000