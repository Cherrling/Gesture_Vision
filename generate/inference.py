import json
from typing import List

import torch
from transformers import AutoTokenizer

from model import SLTModel


def generate_video(gloss_seq: List[str]):
    with open("data/CSL/dictionary.txt") as f:
        lines = f.readlines()
        lines = map(lambda x: x.strip().split(" "), lines)
        lines = filter(lambda x: len(x) == 2, lines)
        lines = map(lambda x: (x[0].strip(), x[1].strip()), lines)
        lines = list(lines)

    gloss_to_idx = {
        gloss: i
        for i, gloss in lines
    }

    frame_ids = [gloss_to_idx[gloss] for gloss in gloss_seq if gloss in gloss_to_idx]
    print(frame_ids)


def inference(input_sentence: str):
    with open("data/CSL-Daily/gloss_vocabs.json", "r") as f:
        gloss_vocabs = json.load(f)
    token_to_gloss = {
        i: gloss for gloss, i in gloss_vocabs.items()
    }

    model = SLTModel.load_from_checkpoint(
        "lightning_logs/version_10/checkpoints/epoch=99-step=32300.ckpt",
        map_location="cpu",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    results = tokenizer(
        input_sentence,
        return_tensors="pt",
    )
    input_ids = results["input_ids"]

    with torch.no_grad():
        gloss_ids = model.generate(input_ids)[0].tolist()

    gloss_seq = [
        token_to_gloss[gloss_id] for gloss_id in gloss_ids if gloss_id in token_to_gloss
    ]

    print("input sentence:\t\t", input_sentence)
    print("translated sentence:\t", " ".join(gloss_seq))

    generate_video(gloss_seq)


if __name__ == "__main__":
    inference("今晚吃什么")
