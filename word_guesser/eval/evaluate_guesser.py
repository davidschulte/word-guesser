import numpy as np
from word_guesser import WordGuesser, SimulatedGame, QdrantWordGuesser, InMemoryWordGuesser
from word_guesser.utils import read_word_frequencies
from typing import Union, Optional
from pathlib import Path
from tqdm.auto import tqdm
from time import perf_counter
import matplotlib.pyplot as plt
import pickle

vocab_file_path = Path("../../data/glove.6B/glove.6B.50d.txt")
vocab_limit = 10_000

word_freq_file_path = Path("../../data/unigram_freq.csv")

num_tries = 5
num_words = 250


def evaluate_guesser(
        guesser: WordGuesser,
        game: SimulatedGame,
        target: Optional[Union[str, int]],
        seed: Optional[int] = None
) -> int:
    state = np.random.get_state()
    if seed:
        np.random.seed(seed)

    game.pick_target(target)
    num_guesses = guesser.play(game, verbose=0)

    np.random.set_state(state)

    return num_guesses


def main():
    game = SimulatedGame(vocab_file_path, vocab_limit=vocab_limit)

    guessers = {
        "InMemoryGuesser": InMemoryWordGuesser(vocab_file_path, vocab_limit=vocab_limit, scoring_threshold=0.5),
        "QdrantGuesser": QdrantWordGuesser(vocab_file_path, vector_dim=50, vocab_limit=vocab_limit),
    }

    results = {}
    for guesser_name in guessers:
        results[guesser_name] = {
            "guess_means": [],
            "guess_stds": [],
            "time_means": [],
            "time_stds": []
        }

    word_frequencies = read_word_frequencies(word_freq_file_path)
    for guesser_name, guesser in guessers.items():
        for target in tqdm(list(word_frequencies.keys())[:num_words], desc=f"Evaluating {guesser_name}", unit="target"):
            if target not in game.words:
                continue
            guess_counts = []
            times = []
            for seed in range(42, 42 + num_tries):
                start_time = perf_counter()
                num_guesses = evaluate_guesser(guesser, game, target=target, seed=seed)
                end_time = perf_counter()

                guess_counts.append(num_guesses)
                times.append(end_time - start_time)

            guess_counts = np.array(guess_counts)
            times = np.array(times)

            results[guesser_name]["guess_means"].append(guess_counts.mean())
            results[guesser_name]["guess_stds"].append(guess_counts.std())
            results[guesser_name]["time_means"].append(times.mean())
            results[guesser_name]["time_stds"].append(times.std())

    with open("results.pkl", "wb") as f:
        pickle.dump(results, f)

    with open("results.pkl", "rb") as f:
        results = pickle.load(f)

    for metric in ("guess", "time"):
        for statistic in ("means", "stds"):
            plt.figure(figsize=(15,10))
            for guesser_name in guessers:
                guesser_results = results[guesser_name]
                plt.hist(guesser_results[f"{metric}_{statistic}"], bins=100, alpha=0.6, label=guesser_name)
            plt.legend()

            statistic_str = "Average" if statistic == "means" else "Standard deviation of"
            metric_str = "word guesses" if metric == "guess" else "time"

            plt.title(f"{statistic_str} {metric_str} for {num_words} most common words")
            plt.xlabel("# guesses" if metric == "guess" else "s")
            plt.ylabel("occurrences")
            plt.savefig(f"{metric}_{statistic}_top_{num_words}.png")

    for guesser_name in guessers:
        print(f"{guesser_name} Avg. number of guesses: {np.array(results[guesser_name]['guess_means']).mean():2f}")
        print(f"{guesser_name} Avg. time: {np.array(results[guesser_name]['time_means']).mean():2f}")


if __name__ == "__main__":
    main()
