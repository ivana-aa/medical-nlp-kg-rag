"""Dataset utilities for BIO-format Chinese medical NER data."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


def read_bio_file(path: str | Path) -> tuple[list[list[str]], list[list[str]]]:
    """Read a BIO file.

    Expected format:

    ``字 标签``

    Blank lines split sentences. Lines with only one token are treated as
    ``O`` labels, which is convenient for quick manual examples.
    """
    path = Path(path)
    sentences: list[list[str]] = []
    labels: list[list[str]] = []
    current_tokens: list[str] = []
    current_labels: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_labels)
                    current_tokens = []
                    current_labels = []
                continue

            parts = line.split()
            token = parts[0]
            label = parts[-1] if len(parts) > 1 else "O"
            current_tokens.append(token)
            current_labels.append(label)

    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_labels)

    return sentences, labels


class MedicalNERDataset(Dataset):
    """PyTorch dataset for character-level BIO NER examples."""

    def __init__(
        self,
        tokens: Sequence[Sequence[str]],
        labels: Sequence[Sequence[str]],
        tokenizer,
        label2id: dict[str, int],
        max_length: int = 128,
    ) -> None:
        self.tokens = list(tokens)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        chars = list(self.tokens[idx])
        label_names = list(self.labels[idx])

        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels: list[int] = []
        previous_word_id = None
        for word_id in word_ids:
            if word_id is None:
                aligned_labels.append(-100)
            elif word_id != previous_word_id:
                aligned_labels.append(self.label2id[label_names[word_id]])
            else:
                # Ignore extra word pieces. Demo data is character-level, so
                # this mainly protects future custom datasets.
                aligned_labels.append(-100)
            previous_word_id = word_id

        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = torch.tensor(aligned_labels, dtype=torch.long)
        return item
