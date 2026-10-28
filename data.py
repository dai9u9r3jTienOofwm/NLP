"""Đọc IMDB gốc; tokenizer và vocabulary chỉ học từ tập train."""

import html
import re
import shutil
import tarfile
from collections import Counter
from http.client import HTTPException
from pathlib import Path
from urllib.request import urlopen

import torch
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

URLS = (
    "https://s3.amazonaws.com/fast-ai-nlp/imdb.tgz",
    "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
)


def download_imdb(archive):
    partial = archive.with_suffix(".part")
    for url in URLS:
        print(f"Downloading IMDB from {url}", flush=True)
        try:
            with urlopen(url, timeout=30) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out)
                expected = response.headers.get("Content-Length")
            if expected and partial.stat().st_size != int(expected):
                raise OSError("Incomplete download")
            partial.replace(archive)
            return
        except (OSError, HTTPException) as error:
            partial.unlink(missing_ok=True)
            print(f"Download failed: {error}", flush=True)
            last_error = error
    raise RuntimeError(
        "Cannot download IMDB from either source. Check your connection, or download "
        f"{URLS[0]} in a browser and save it as {archive}, then rerun."
    ) from last_error


def tokenize(text):
    text = re.sub(r"<[^>]+>", " ", html.unescape(text))
    return re.findall(r"[a-z]+(?:'[a-z]+)?|[!?]", text.lower())


def load_imdb(data_dir, limit=0, seed=42):
    root = Path(data_dir) / "aclImdb"
    folders = [root / split / label for split in ("train", "test")
               for label in ("neg", "pos")]
    if not all(len(list(folder.glob("*.txt"))) == 12500 for folder in folders):
        archive = Path(data_dir) / "aclImdb_v1.tar.gz"
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            download_imdb(archive)
        print("Extracting labeled IMDB reviews...", flush=True)
        # Chỉ ghi review có nhãn; không extract đường dẫn tùy ý từ archive.
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                parts = member.name.split("/")
                if (member.isfile() and len(parts) == 4 and parts[0] in ("aclImdb", "imdb")
                        and parts[1] in ("train", "test")
                        and parts[2] in ("neg", "pos")
                        and re.fullmatch(r"\d+_\d+\.txt", parts[3])):
                    dest = root.joinpath(*parts[1:])
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, dest.open("wb") as out:
                        shutil.copyfileobj(src, out)
    if not all(len(list(folder.glob("*.txt"))) == 12500 for folder in folders):
        raise RuntimeError("Expected 50,000 IMDB reviews. Check/remove a corrupt archive and retry.")

    def read(split):
        texts, labels = [], []
        for label, name in enumerate(("neg", "pos")):
            for path in sorted((root / split / name).glob("*.txt")):
                texts.append(path.read_text(encoding="utf-8"))
                labels.append(label)
        if limit and limit < len(texts):
            texts, _, labels, _ = train_test_split(
                texts, labels, train_size=limit, stratify=labels, random_state=seed)
        return texts, labels

    texts, labels = read("train")
    x_train, x_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=seed)
    return (x_train, y_train), (x_val, y_val), read("test")


def build_vocab(texts, size):
    counts = Counter(token for text in texts for token in tokenize(text))
    return {"<pad>": 0, "<unk>": 1,
            **{word: i + 2 for i, (word, _) in enumerate(counts.most_common(size - 2))}}


def collate(batch):
    sequences, labels = zip(*batch)
    return (pad_sequence(sequences, batch_first=True),
            torch.tensor([len(x) for x in sequences]), torch.tensor(labels))


def make_loader(split, vocab, max_len, batch_size, shuffle=False):
    texts, labels = split
    sequences = [torch.tensor([vocab.get(t, 1) for t in tokenize(text)[:max_len]] or [1])
                 for text in texts]
    return DataLoader(list(zip(sequences, labels)), batch_size=batch_size,
                      shuffle=shuffle, collate_fn=collate)
