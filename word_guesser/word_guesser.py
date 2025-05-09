import numpy as np
from abc import abstractmethod

from qdrant_client import QdrantClient, models
from pathlib import Path
import math
from tqdm.auto import tqdm
from typing import Optional

from word_guesser.word_guessing_game import WordGuessingGame
from word_guesser.utils import read_vocab, read_glove_line, read_word_frequencies


class WordGuesser:

    def __init__(self):
        self.guesses = []
        self.guess_ids = []
        self.guess_ranks = []
        self.vocab_size = -1

    def reset_guesses(self):
        self.guesses = []
        self.guess_ids = []
        self.guess_ranks = []

    @abstractmethod
    def make_guess(self):
        pass

    def play(self, game: WordGuessingGame, max_num_guesses: int = 100) -> int:
        self.reset_guesses()
        for num_guess in range(max_num_guesses):
            guess = self.make_guess()
            guess_rank = game.rank_guess(guess)
            self.guess_ranks.append(guess_rank)

            if guess_rank == 0:
                print(f"Guessed the word {guess} after {num_guess} guesses!")
                return num_guess
            
        print(f"Couldn't find the word {game.tell_target()} with {max_num_guesses} tries. Last guess was {self.guesses[-1]}.")
        return -1


class QdrantWordGuesser(WordGuesser):

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
            batch = lines[batch_i * batch_size : min((batch_i + 1) * batch_size, len(lines))]
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

            prev_other_guess_ids = [guess_id for idx, guess_id in enumerate(self.guess_ids) if idx != prev_best_guess_idx]

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

    def __init__(self, vocab_file_path: Path):
        super().__init__()
        self.initialize_vocab(vocab_file_path)
        self.scored_similarities = None


    def reset_guesses(self):
        super().reset_guesses()
        self.scored_similarities = np.zeros(self.vocab_size)

    def initialize_vocab(self, vocab_file_path: Path, word_freq_file_path: Optional[Path] = None):
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
            a=1

    def _updated_scored_similarities(self, scoring_threshold: float = 0.9999):
        if len(self.guesses) == 0:
            return

        last_rank_score = 1 - (self.guess_ranks[-1] / self.vocab_size)
        last_score = np.exp(last_rank_score) - np.exp(scoring_threshold)
        self.scored_similarities += self.embeddings @ self.embeddings[self.guess_ids[-1]].T * last_score
        self.scored_similarities[self.guess_ids[-1]] = 0


    def make_guess(self, scoring_threshold: float = 0.9999) -> str:
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

    def make_guess(self) -> str:
        return input("Make a guess: ")
