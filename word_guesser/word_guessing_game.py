from pathlib import Path
import numpy as np
from .utils import read_vocab, read_glove_line
from typing import Optional, Union
import difflib
import requests
from datetime import date, datetime
from abc import ABC, abstractmethod

CONTEXTO_BASE_URL = "https://api.contexto.me/machado"
LANGUAGE = "en"
CONTEXTO_START_DATE = "2022-09-18"

class WordGuessingGame(ABC):
    """
    Abstract class for WordGuessingGames
    """

    @abstractmethod
    def rank_guess(self, guess: str) -> int:
        """
        Ranks a guess

        Args:
            guess: The guess

        Returns:
            The rank of the guess. If the word is not in the game vocabulary, -1 is returned.
        """
        pass

    @abstractmethod
    def tell_target(self) -> str:
        """
        Tells the target word

        Returns:
            The target word
        """
        pass

class SimulatedGame(WordGuessingGame):

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
            The rank of the guess. If the word is not in the game vocabulary, -1 is returned.
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


class ContextoGame(WordGuessingGame):

    def __init__(self, game_id: Optional[int] = None):
        max_possible_id = (date.today() - datetime.strptime("2022-09-18", "%Y-%m-%d").date()).days

        if game_id is not None :
            if not 0 <= game_id <= max_possible_id:
                raise ValueError((f"Could not initialize a ContextoGame with ID {game_id}. "
                                  f"Please choose an idea between 0 and {max_possible_id}."))
            self.game_id = game_id
        else:
            self.game_id = np.random.randint(0, max_possible_id + 1)
            print(self.game_id, type(self.game_id))

        # print(f"{CONTEXTO_BASE_URL}/{LANGUAGE}/giveup/{self.game_id}")
        target_word_response = requests.get(f"{CONTEXTO_BASE_URL}/{LANGUAGE}/giveup/{self.game_id}")

        if target_word_response.status_code == 200:
            self.target_word = target_word_response.json()["word"]
        else:
            print(target_word_response.status_code)
            raise Exception(f"{target_word_response.status_code}: {target_word_response.text}")


    def rank_guess(self, guess: str) -> int:
        """
        Ranks a guess

        Args:
            guess: The guess

        Returns:
            The rank of the guess. If the word is not in the game vocabulary, -1 is returned.
        """
        guess_response = requests.get(f"{CONTEXTO_BASE_URL}/{LANGUAGE}/game/{self.game_id}/{guess}")
        if guess_response.status_code == 200:
            return guess_response.json()["distance"]

        return -1

    def tell_target(self) -> str:
        """
        Tells the target word

        Returns:
            The target word
        """
        return self.target_word
