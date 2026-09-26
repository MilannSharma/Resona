"""
Resona Hinglish Engine
High-fidelity code-switching transducer converting Roman Hinglish into Devanagari Hindi
while strictly protecting English UI, software, and technical terminology.
"""

import os
import re
import json
from typing import Dict, List, Set, Tuple, Optional

class ResonaHinglishEngine:
    """
    Decoupled Hinglish code-switching engine utilizing:
      1. Compound Phrases (matched longest-first)
      2. Single-Word Transliteration Dictionary
      3. Protected English UI/Tech Term Whitelist
    """

    def __init__(self, vocab_dir: Optional[str] = None):
        if not vocab_dir:
            pkg_vocab = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vocab"))
            repo_vocab = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vocab"))
            if os.path.isdir(pkg_vocab):
                vocab_dir = pkg_vocab
            elif os.path.isdir(repo_vocab):
                vocab_dir = repo_vocab
            else:
                vocab_dir = pkg_vocab

        self.vocab_dir = vocab_dir
        self.compound_phrases: List[Tuple[str, str, re.Pattern]] = []
        self.words_map: Dict[str, str] = {}
        self.whitelist: Set[str] = set()

        self._load_vocab()

    def _load_vocab(self):
        # 1. Phrases
        p_path = os.path.join(self.vocab_dir, "compound_phrases.json")
        if os.path.isfile(p_path):
            with open(p_path, "r", encoding="utf-8") as f:
                raw_phrases = json.load(f)
            # Compile regex for each phrase with word boundaries
            for item in raw_phrases:
                phrase = item["phrase"].lower()
                dev = item["devanagari"]
                pattern = re.compile(r'\b' + re.escape(phrase) + r'\b', re.IGNORECASE)
                self.compound_phrases.append((phrase, dev, pattern))

        # 2. Whitelist
        w_path = os.path.join(self.vocab_dir, "tech_whitelist.json")
        if os.path.isfile(w_path):
            with open(w_path, "r", encoding="utf-8") as f:
                raw_whitelist = json.load(f)
            self.whitelist = {w.lower().strip() for w in raw_whitelist}

        # 3. Words Lexicon
        l_path = os.path.join(self.vocab_dir, "hinglish_lexicon.json")
        if os.path.isfile(l_path):
            with open(l_path, "r", encoding="utf-8") as f:
                self.words_map = json.load(f)

    def normalize(self, text: str) -> str:
        """
        Translates Hinglish text into Devanagari Hindi while preserving English terms.
        """
        if not text:
            return ""

        # Step 1: Match and replace compound phrases longest-first
        processed_text = text
        for phrase, dev, pattern in self.compound_phrases:
            processed_text = pattern.sub(dev, processed_text)

        # Step 2: Tokenize and translate single words with whitelist guard
        tokens = re.findall(r'[a-zA-Z0-9_\-\u0900-\u097F]+|[^\w\s]|\s+', processed_text)
        result = []

        for token in tokens:
            lower = token.lower()
            # If word is in tech whitelist, protect it in English
            if lower in self.whitelist:
                result.append(token)
            # If word is in Hinglish words map, transliterate to Devanagari
            elif lower in self.words_map:
                result.append(self.words_map[lower])
            else:
                result.append(token)

        return "".join(result)
