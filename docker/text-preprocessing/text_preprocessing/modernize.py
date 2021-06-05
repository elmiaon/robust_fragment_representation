#!/usr/bin/env python3
"""Modernizer wrapper for all languages"""

from typing import Dict


class modernizer:
    """Modernizes supported languages
    Emulates a basic dict interface with dict[key] and .get method"""

    language_dict: Dict[str, str]

    @classmethod
    def __init__(cls, language: str):
        if language == "french":
            from .lang.fr_dict import french_dict

            cls.language_dict = french_dict
        elif language == "english":
            from .lang.en_dict import english_dict

            cls.language_dict = english_dict

    def __call__(self, word: str):
        return self.language_dict.get(word, word)

    def get(self, word):
        """Retrieve modernizer word"""
        return self(word)

    def __getitem__(self, item):
        """Retrieve modernized word"""
        return self(item)
