from abc import abstractmethod
from typing import Optional
from pathlib import Path
import math

import numpy as np
from qdrant_client import QdrantClient, models
from tqdm.auto import tqdm

from word_guesser.word_guessing_game import WordGuessingGame
from word_guesser.utils import read_vocab, read_glove_line, read_word_frequencies


class WordGuesser:

    def __init__(self):
        self.guesses = []
        self.guess_ids = []
        self.guess_ranks = []
        self.vocab_size = -1

    def _reset_guesses(self):
        self.guesses = []
        self.guess_ids = []
        self.guess_ranks = []

    @abstractmethod
    def make_guess(self) -> str:
        """
        Makes the next guess

        Returns:
            The next guess
        """
        pass

    def play(self, game: WordGuessingGame, max_num_guesses: Optional[int] = None, verbose: int = 1) -> int:
        """
        Plays a game

        Args:
            game: The game to be played
            max_num_guesses: The maximum number of allowed guesses. If set to -1, there is no limit
            verbose: 0 suppresses all console logs, 1 prints the outcome, 2 shows the ranking of each guess

        Returns:
            The number of guesses used or -1 if the maximum number of guesses is surpassed
        """
        self._reset_guesses()
        while not max_num_guesses or len(self.guesses) <= max_num_guesses - 1:
            guess = self.make_guess()
            guess_rank = game.rank_guess(guess)
            if guess_rank < 0:
                continue
            self.guesses.append(guess)
            self.guess_ranks.append(guess_rank)

            if guess_rank == 0:
                if verbose > 0:
                    print(f"Guessed the word {guess} after {len(self.guesses)} guesses!")
                return len(self.guesses)

            if verbose > 1:
                print(f"Rank {guess_rank:<6}: {guess}")

        print((f"Couldn't find the word {game.tell_target()} with {max_num_guesses} tries."
               f"Last guess was {self.guesses[-1]}."))
        return -1


class QdrantWordGuesser(WordGuesser):
    """
    The QdrantWordGuesser uses Qdrant to store its vocabulary and embeddings. To select next guesses it uses
    Qdrant's recommendation function. It always chooses the best previous guess as a positive example and the other
    guesses as negative examples.
    For more details see: https://qdrant.tech/documentation/concepts/explore/#average-vector-strategy
    """

    def __init__(
            self,
            vocab_file_path: Path,
            vector_dim: int,
            qdrant_host_url: str = "http://localhost:6333",
            collection_name: Optional[str] = None,
    ):
        super(QdrantWordGuesser, self).__init__()
        self.vector_dim = vector_dim
        self.client = QdrantClient(url=qdrant_host_url)
        self.collection_name = collection_name

        self.initialize_vocab(vocab_file_path)

    def initialize_vocab(self, vocab_file_path: Path, overwrite_collection: bool = False, batch_size: int = 1024):
        """
        Initializes the vocabulary of the guesser

        Args:
            vocab_file_path: The path to a file with GloVe embeddings
            overwrite_collection: If set to True, the collection will be overwritten
            batch_size: The batch size for adding points to the collection

        Returns:

        """
        if self.collection_name is None:
            self.collection_name = vocab_file_path.stem

        if overwrite_collection:
            self.client.delete_collection(collection_name=self.collection_name)
        if not self.client.collection_exists(collection_name=self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=self.vector_dim, distance=models.Distance.COSINE),
            )

        lines = read_vocab(vocab_file_path)
        self.vocab_size = len(lines)

        collection_count = self.client.count(collection_name=self.collection_name).count

        if collection_count == len(lines) and not overwrite_collection:
            return

        num_batches = math.ceil(len(lines) / batch_size)
        for batch_i in tqdm(range(num_batches)):
            batch = lines[batch_i * batch_size: min((batch_i + 1) * batch_size, len(lines))]
            batch_lines_read = [read_glove_line(line) for line in batch]
            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    models.PointStruct(id=batch_i * batch_size + i, vector=embedding, payload={"word": word})
                    for i, (word, embedding) in enumerate(batch_lines_read)
                ],
            )

    def make_guess(self, strategy: models.RecommendStrategy = models.RecommendStrategy.AVERAGE_VECTOR) -> str:
        """
        Makes the next guess

        Args:
            strategy: A recommendation strategy for the next guess

        Returns:
            The next guess
        """
        if len(self.guesses) < 2:
            guess_id = np.random.randint(0, self.vocab_size)

            guess_response = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[guess_id],
            )
            guess = guess_response[0].payload["word"]

        else:
            prev_best_guess_idx = np.argmin(self.guess_ranks)
            prev_best_guess_id = self.guess_ids[prev_best_guess_idx]

            prev_other_guess_ids = [id_ for idx, id_ in enumerate(self.guess_ids) if idx != prev_best_guess_idx]

            query_result = self.client.query_points(
                collection_name="test_collection",
                query=models.RecommendQuery(
                    recommend=models.RecommendInput(
                        positive=[prev_best_guess_id],
                        negative=prev_other_guess_ids,
                        strategy=strategy,
                    )
                ),
                limit=1,
            )
            
            guess = query_result.points[0].payload["word"]
            guess_id = query_result.points[0].id

        self.guesses.append(guess)
        self.guess_ids.append(guess_id)

        return guess


class InMemoryWordGuesser(WordGuesser):
    """
    The InMemoryWordGuesser stores its vocabulary and embeddings in memory. When selecting next guesses it favors
    words that are similar to good previous guesses and different from bad previous guesses. This is handled using the
    scoring threshold. For example, a scoring threshold of 0.9999 declares the best possible 0.1% of guesses as good and
    the others as bad guesses. In contrast to the QdrantWordGuesser, this weighting is not binary but continuously
    calculated with an exponential score function.
    The InMemoryWordGuesser further allows favoring commonly used words if supplied with frequencies.
    """

    def __init__(
            self,
            vocab_file_path: Path,
            word_freq_file_path: Optional[Path] = None,
            scoring_threshold: float = 0.9999
    ):
        """

        Args:
            vocab_file_path: The file path to a file with GloVe embeddings
            word_freq_file_path: The file path to file with word frequencies
            scoring_threshold: The scoring threshold (see class documentation)
        """
        super().__init__()
        self.words: Optional[list[str]] = None
        self.word2idx: Optional[dict[str, int]] = None
        self.embeddings: Optional[np.array] = None
        self.word_frequencies: Optional[np.array] = None
        self.scoring_threshold = scoring_threshold

        self.initialize_vocab(vocab_file_path, word_freq_file_path=word_freq_file_path)
        self.scored_similarities = None

    def _reset_guesses(self):
        super()._reset_guesses()
        self.scored_similarities = np.zeros(self.vocab_size)

    def initialize_vocab(self, vocab_file_path: Path, word_freq_file_path: Optional[Path] = None):
        """
        Initializes the vocabulary of the guesser

        Args:
            vocab_file_path: The path to a file with GloVe embeddings
            word_freq_file_path: The path to a file with word frequencies

        Returns:

        """
        lines = read_vocab(vocab_file_path)

        words, embeddings = zip(*[read_glove_line(l) for l in lines])

        self.words = words
        self.word2idx = {word: idx for idx, word in enumerate(words)}
        self.embeddings = np.array(embeddings)
        self.embeddings /= np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.vocab_size = len(words)

        if word_freq_file_path is not None:
            word_frequencies = read_word_frequencies(word_freq_file_path)
            self.word_frequencies = np.array([word_frequencies.get(word, 0) for word in self.words])

    def _updated_scored_similarities(self, scoring_threshold: float = 0.9999):
        """
        This helper function enables efficient tracking of scores using previous guesses and their ranks.
        It is called before making the next guess and updates the scores for next guesses using the previous rank.

        Args:
            scoring_threshold: The scoring threshold defines which previous guesses serve as negative and
                which previous guesses serve as positive examples. For example, with a scoring threshold of 0.9999
                all the previous guesses whose rank is in the 0.1% of possible ranks are positive example; and the
                others are negative examples.

        Returns:

        """
        if len(self.guesses) == 0:
            return

        # Compute the normalized rank score of the last rank (1 = perfect guess, 0 = worst possible guess)
        last_rank_score = 1 - (self.guess_ranks[-1] / self.vocab_size)

        # The scoring function of the last guess. It gets a positive score if the guess is very close to the best
        # possible guess and a negative one, when it isn't.
        last_score = np.exp(last_rank_score) - np.exp(scoring_threshold)

        # We update the scored similarities for next guesses.
        # If the last guess was good, it got a positive score, and the scored similarities for similar guesses increase.
        # Otherwise, the scored similarities for similar guesses decrease, such that next guess will be different.
        self.scored_similarities += self.embeddings @ self.embeddings[self.guess_ids[-1]].T * last_score
        self.scored_similarities[self.guess_ids[-1]] = 0

    def make_guess(self, scoring_threshold: Optional[float] = None) -> str:
        """
        Makes the next guess. The first two guesses are random. After that, the next guess is aimed to be far away
        from bad guesses and close to good guesses. In contrast to the QdrantGuesser, this is done using
        a continuous score.

        Args:
            scoring_threshold: The scoring threshold defines which previous guesses serve as negative and
                which previous guesses serve as positive examples. For example, with a scoring threshold of 0.9999
                all the previous guesses whose rank is in the 0.1% of possible ranks are positive example; and the
                others are negative examples.

        Returns:
            The next guess
        """
        if not scoring_threshold:
            scoring_threshold = self.scoring_threshold

        self._updated_scored_similarities(scoring_threshold=scoring_threshold)

        if len(self.guesses) < 2:
            guess_id = np.random.randint(0, self.vocab_size)
            guess = self.words[guess_id]
        else:
            curr_scored_similarities = self.scored_similarities
            if self.word_frequencies is not None:
                curr_score_pos_idxs = curr_scored_similarities > 0
                curr_scored_similarities[curr_score_pos_idxs] *= self.word_frequencies[curr_score_pos_idxs]

            guess_id = curr_scored_similarities.argmax()
            guess = self.words[guess_id]
        
        self.guesses.append(guess)
        self.guess_ids.append(guess_id)

        return guess


class HumanWordGuesser(WordGuesser):
    """
    A guesser class with which players can play the game using the console
    """

    def make_guess(self) -> str:
        """
        Makes the next guess

        Returns:
            The next guess
        """
        while True:
            guess = input("Make a guess: ")
            if guess in self.guesses:
                print(f"You already guessed {guess}. It's rank {self.guess_ranks[self.guesses.index(guess)]}.")
            else:
                return guess
