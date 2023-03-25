import pickle
import json
from functools import partial

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class CSLDailyImpl(Dataset):
    def __init__(
        self,
        samples,
        gloss_vocabs
    ):
        super().__init__()

        self.samples = samples
        self.gloss_vocabs = gloss_vocabs
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sentence, gloss_seq = sample["sentence"], sample["gloss_seq"]

        results = self.tokenizer(
            sentence,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = results["input_ids"].squeeze(0)
        attention_mask = results["attention_mask"].squeeze(0)

        gloss_ids = [101] + [
            self.gloss_vocabs[gloss]
            for gloss in gloss_seq
        ] + [102]
        if len(gloss_ids) < 64:
            gloss_ids += [0] * (64 - len(gloss_ids))
        gloss_ids = torch.as_tensor(gloss_ids, dtype=torch.int64)

        return input_ids, attention_mask, gloss_ids


class CSLDaily(pl.LightningDataModule):
    def __init__(
        self,
        train_batch_size: int = 64,
        val_batch_size: int = 64,
        test_batch_size: int = 64,
        num_workers: int = 8,
    ) -> None:
        super().__init__()

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

    def setup(self, stage: str) -> None:
        samples = {}
        with open("data/CSL-Daily/sentence_label/csl2020ct_v2.pkl", "rb") as f:
            annos = pickle.load(f)
            gloss_vocabs = {
                vocab: i + 106
                for i, vocab in enumerate(annos["gloss_map"])
            }
            with open("data/CSL-Daily/gloss_vocabs.json", "w") as f:
                json.dump(gloss_vocabs, f, ensure_ascii=False)

            for info in annos["info"]:
                samples[info["name"]] = {
                    "gloss_seq": info["label_gloss"],
                    "sentence": "".join(info["label_char"]),
                    "split": "train",
                }

        with open("data/CSL-Daily/sentence_label/split_1.txt") as f:
            _ = f.readline()
            while True:
                line = f.readline()
                if not line:
                    break
                name, split = line.strip().split("|")
                if name in samples:
                    samples[name]["split"] = split

        self.train_samples = [
            sample for sample in samples.values() if sample["split"] == "train"
        ]
        self.val_samples = [
            sample for sample in samples.values() if sample["split"] == "dev"
        ]
        self.test_samples = [
            sample for sample in samples.values() if sample["split"] == "dev"
        ]

        self.train_dataset = CSLDailyImpl(self.train_samples, gloss_vocabs)
        self.val_dataset = CSLDailyImpl(self.val_samples, gloss_vocabs)
        self.test_dataset = CSLDailyImpl(self.test_samples, gloss_vocabs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
