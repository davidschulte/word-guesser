from pathlib import Path
import pandas as pd
import numpy as np

def read_vocab(vocab_file_path: Path) -> list[str]:
    with open(vocab_file_path, "r") as f:
        lines = f.readlines()

    return lines


def read_glove_line(line: str) -> tuple[str, list[float]]:
    split_line = line.split()

    word = split_line[0]
    embedding = [float(num) for num in split_line[1:]]

    return word, embedding


def read_word_frequencies(word_freq_file_path: Path) -> dict[str, float]:
    df = pd.read_csv(word_freq_file_path)

    for required_col in ("word", "count"):
        if required_col not in df.columns:
            raise ValueError(f"The frequency file must contain the column {required_col}.")

    df["freq"] = np.log10(df["count"])
    df["freq"] /= df["freq"].max()

    return df.set_index("word")["freq"].to_dict()
