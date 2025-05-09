from pathlib import Path
import numpy as np
from .utils import read_vocab, read_glove_line
from typing import Optional, Union
import difflib


class WordGuessingGame:

    def __init__(self, vocab_file_path: Path, vocab_limit: Optional[int] = None):
        self.words = None
        self.word2idx = None
        self.embeddings = None
        self.vocab_size = None
        self.vocab_limit = vocab_limit

        self.initialize_vocab(vocab_file_path)

        self.target_word = None
        self.target_id = None
        self.word_ranking = None

    def initialize_vocab(self, vocab_file_path: Path):
        """
        Initializes the vocabulary of the game

        Args:
            vocab_file_path: The path to a file with GloVe embeddings

        Returns:

        """
        lines = read_vocab(vocab_file_path)
        if self.vocab_limit:
            lines = lines[:self.vocab_limit]

        words, embeddings = zip(*[read_glove_line(l) for l in lines])
        embeddings = np.array(embeddings)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.words = words
        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.embeddings = embeddings
        self.vocab_size = len(words)

    def pick_target(self, target: Optional[Union[str, int]] = None):
        """
        Sets a target word for the game

        Args:
            target: The target word can either be passed as the word or as the id in the vocabulary

        Returns:

        """
        if isinstance(target, str):
            if target not in self.words:
                raise ValueError(f"{target} is not in the vocabulary.")
            
            self.target_word = target
            self.target_id = self.word2idx[target]
            
        else:
            if not target:
                target = np.random.randint(0, self.vocab_size)

            if isinstance(target, int):
                if not 0 <= target < len(self.words):
                    raise ValueError(f"ID {target} is not in the vocabulary.")

                self.target_word = self.words[target]
                self.target_id = target

        self.word_ranking = (-self.embeddings @ self.embeddings[self.target_id].T).argsort()

    def rank_guess(self, guess: str, suggest_vocabulary_word: bool = True) -> int:
        """
        Ranks a guess

        Args:
            guess: The guess
            suggest_vocabulary_word: If set to True and the word is not in the game vocabulary,
                a similar word is suggested.

        Returns:

        """
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
        """
        Tells the target word

        Returns:
            The target word
        """
        return self.target_word
