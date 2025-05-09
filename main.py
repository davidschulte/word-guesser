from word_guesser import WordGuessingGame, InMemoryWordGuesser, QdrantWordGuesser
from pathlib import Path

vocab_file_path = Path("glove.6B/glove.6B.50d.txt")
word_frequency_file_path = Path("data/unigram_freq.csv")
num_targets = 10
num_games_per_target = 3
def main():

    game = WordGuessingGame(vocab_file_path=vocab_file_path)
    in_memory_player = InMemoryWordGuesser(
        vocab_file_path=vocab_file_path,
        #word_freq_file_path=word_frequency_file_path
    )
    qdrant_player = QdrantWordGuesser(vocab_file_path=vocab_file_path, vector_dim=50)
    human_player = HumanWordGuesser()

    for _ in range(num_targets):
        game.pick_target()

        for _ in range(num_games_per_target):
            player.play(game)


if __name__ == "__main__":
    main()