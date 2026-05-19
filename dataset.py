from collections import Counter
from typing import Iterable, Optional

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Vocab:
    def __init__(self, tokens: Iterable[str], unk_token: str = "<unk>"):
        self.itos = list(tokens)
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}
        self.unk_token = unk_token
        self.unk_idx = self.stoi.get(unk_token, 0)

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.unk_idx)

    def lookup_token(self, idx: int) -> str:
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return self.unk_token


class Multi30kDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocab] = None,
        tgt_vocab: Optional[Vocab] = None,
        min_freq: int = 1,
    ):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        from datasets import load_dataset
        import spacy

        split_alias = {
            "train": "train",
            "val": "validation",
            "valid": "validation",
            "validation": "validation",
            "test": "test",
        }
        split = split_alias.get(split, split)
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Unsupported split: {split}")

        self.split = split
        self.min_freq = min_freq

        self.special_tokens = ["<unk>", "<pad>", "<sos>", "<eos>"]

        self.dataset = load_dataset("bentrevett/multi30k", split=self.split)

        try:
            self.de_tokenizer = spacy.load("de_core_news_sm")
        except Exception:
            self.de_tokenizer = spacy.blank("de")

        try:
            self.en_tokenizer = spacy.load("en_core_web_sm")
        except Exception:
            self.en_tokenizer = spacy.blank("en")

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        self.data = self.process_data()

    def _tokenize_de(self, text: str) -> list[str]:
        return [tok.text.lower() for tok in self.de_tokenizer(text)]

    def _tokenize_en(self, text: str) -> list[str]:
        return [tok.text.lower() for tok in self.en_tokenizer(text)]

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        src_counter = Counter()
        tgt_counter = Counter()

        for sample in self.dataset:
            src_counter.update(self._tokenize_de(sample["de"]))
            tgt_counter.update(self._tokenize_en(sample["en"]))

        def make_vocab(counter: Counter) -> Vocab:
            tokens = list(self.special_tokens)
            for tok, freq in counter.most_common():
                if freq >= self.min_freq and tok not in tokens:
                    tokens.append(tok)
            return Vocab(tokens)

        src_vocab = make_vocab(src_counter)
        tgt_vocab = make_vocab(tgt_counter)
        return src_vocab, tgt_vocab

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary.
        """
        src_sos = self.src_vocab["<sos>"]
        src_eos = self.src_vocab["<eos>"]
        tgt_sos = self.tgt_vocab["<sos>"]
        tgt_eos = self.tgt_vocab["<eos>"]

        examples = []
        for sample in self.dataset:
            src_tokens = self._tokenize_de(sample["de"])
            tgt_tokens = self._tokenize_en(sample["en"])

            src_ids = [src_sos] + [self.src_vocab[tok] for tok in src_tokens] + [src_eos]
            tgt_ids = [tgt_sos] + [self.tgt_vocab[tok] for tok in tgt_tokens] + [tgt_eos]

            src_tensor = torch.tensor(src_ids, dtype=torch.long)
            tgt_tensor = torch.tensor(tgt_ids, dtype=torch.long)
            examples.append((src_tensor, tgt_tensor))

        return examples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def collate_fn(self, batch):
        src_batch, tgt_batch = zip(*batch)
        src_pad_idx = self.src_vocab["<pad>"]
        tgt_pad_idx = self.tgt_vocab["<pad>"]

        src_batch = pad_sequence(src_batch, batch_first=True, padding_value=src_pad_idx)
        tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=tgt_pad_idx)
        return src_batch, tgt_batch
