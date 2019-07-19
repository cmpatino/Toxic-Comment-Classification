# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 09:44:49 2019

@author: Bananin
"""

from spellchecker import SpellChecker
import re

def correct_spelling (text):
    # spell-checking tool
    spell = SpellChecker(language="en")

    # no urls
    text = re.sub("(http|www)[^ ]*","", text)
    # no unusual letter repetitions
    text = reduce_lengthening(text)
    # letters only
    text = re.sub("[^a-zA-Z]", " ", text)
    # correct spelling
    text = spell.correction(text)

    return text

def reduce_lengthening(text):
    pattern = re.compile(r"(.)\1{2,}")
    return pattern.sub(r"\1\1", text)
