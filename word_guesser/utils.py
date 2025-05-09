from pathlib import Path
import pandas as pd
import numpy as np


def read_vocab(vocab_file_path: Path) -> list[str]:
    """
    Reads a GloVe vocabulary file and returns its lines

    Args:
        vocab_file_path: The path to a GloVe vocabulary file

    Returns:
        The read line
    """
    with open(vocab_file_path, "r") as f:
        lines = f.readlines()

    return lines


def read_glove_line(line: str) -> tuple[str, list[float]]:
    """
    Reads a GloVe line and splits it into a word and an embedding

    Args:
        line: A GloVe line

    Returns:
        The word and the embedding
    """
    split_line = line.split()

    word = split_line[0]
    embedding = [float(num) for num in split_line[1:]]

    return word, embedding


def read_word_frequencies(word_freq_file_path: Path) -> dict[str, float]:
    """
    Reads a word frequency file. The file must be a csv with the columns "word" and "count".
    Their frequencies get transformed by first applying the natural logarithm to their count, and then dividing their
    count by the maximum count in the file.

    Args:
        word_freq_file_path: A path to word frequency file

    Returns:
        A dictionary with the words as keys and the transformed frequencies as values
    """
    df = pd.read_csv(word_freq_file_path)

    for required_col in ("word", "count"):
        if required_col not in df.columns:
            raise ValueError(f"The frequency file must contain the column {required_col}.")

    df["freq"] = np.log10(df["count"])
    df["freq"] /= df["freq"].max()

    return df.set_index("word")["freq"].to_dict()
