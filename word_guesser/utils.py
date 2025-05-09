from pathlib import Path


def read_vocab(vocab_file_path: Path) -> list[str]:
    with open(vocab_file_path, "r") as f:
        lines = f.readlines()

    return lines


def read_glove_line(line: str) -> tuple[str, list[float]]:
    split_line = line.split()

    word = split_line[0]
    embedding = [float(num) for num in split_line[1:]]

    return word, embedding
