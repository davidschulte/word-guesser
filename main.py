from word_guesser import WordGuessingGame, InMemoryWordGuesser, QdrantWordGuesser, HumanWordGuesser
from pathlib import Path

vocab_file_path = Path("data/glove.6B/glove.6B.50d.txt")
word_frequency_file_path = Path("data/unigram_freq.csv")
vocab_limit = 10_000
num_targets = 10
num_games_per_target = 10


def main():
    game = WordGuessingGame(vocab_file_path=vocab_file_path, vocab_limit=vocab_limit)
    in_memory_player = InMemoryWordGuesser(
        vocab_file_path=vocab_file_path,
        vocab_limit=vocab_limit,
        scoring_threshold=0.5
    )
    qdrant_player = QdrantWordGuesser(vocab_file_path=vocab_file_path, vector_dim=50, vocab_limit=vocab_limit)
    human_player = HumanWordGuesser()

    for _ in range(num_targets):
        game.pick_target()

        for player_name, player in (("InMemory", in_memory_player), ("Qdrant", qdrant_player), ("Human", human_player)):
            print(f"{player_name}:")
            for _ in range(num_games_per_target):
                player.play(game, verbose=(2 if player_name == "Human" else 1))


if __name__ == "__main__":
    main()
