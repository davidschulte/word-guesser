from pathlib import Path
import numpy as np
from .utils import read_vocab, read_glove_line
from typing import Optional
import difflib


class WordGuessingGame:

    def __init__(self, vocab_file_path: Path):
        self.initialize_vocab(vocab_file_path)

        self.target_word = None
        self.target_id = None
        self.word_ranking = None

    def initialize_vocab(self, vocab_file_path: Path):
        lines = read_vocab(vocab_file_path)

        words, embeddings = zip(*[read_glove_line(l) for l in lines])
        embeddings = np.array(embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.words = words
        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.embeddings = embeddings
        self.vocab_size = len(words)

    def pick_target(self, word: Optional[str] = None, id: Optional[int] = None):
        if word and id:
            raise ValueError("You can ony specify a word or a word ID.")

        if word:
            if word not in self.words:
                raise ValueError(f"{word} is not in the vocabulary.")
            
            self.target_word = word
            self.target_id = self.word2idx[word]
            
        else:
            if not id:
                id = np.random.randint(0, self.vocab_size)

            if not 0 <= id < len(self.words):
                raise ValueError(f"ID {id} is not in the vocabulary.")

            self.target_word = self.words[id]
            self.target_id = id

        self.word_ranking = (-self.embeddings @ self.embeddings[self.target_id].T).argsort()

    def rank_guess(self, guess: str, suggest_vocabulary_word: bool = True) -> int:
        if guess not in self.words:
            print(f"The word {guess} is not in the vocabulary.")
            if suggest_vocabulary_word:
                similar_words = difflib.get_close_matches(guess, self.words, n=1)
                if similar_words:
                    print(f"Did you mean {similar_words[0]}?")
            
            return -1
        
        guess_idx = self.word2idx[guess]

        return int(np.argmax(self.word_ranking == guess_idx))

    def tell_target(self) -> str:
        return self.target_word
