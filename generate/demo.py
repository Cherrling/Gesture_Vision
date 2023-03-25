import json
import pickle
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import textdistance
from moviepy.editor import ImageSequenceClip
from transformers import AutoTokenizerv

from model import SLTModel

model, tokenizer = None, None
token_to_gloss = None
videos = None


def get_model_and_tokenizer():
    global model, tokenizer

    if model is None:
        model = SLTModel.load_from_checkpoint(
            "lightning_logs/version_10/checkpoints/epoch=99-step=32300.ckpt",
            map_location="cpu",
        )
        model.eval()

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    return model, tokenizer


def get_token_to_gloss():
    global token_to_gloss

    if token_to_gloss is None:
        with open("data/CSL-Daily/gloss_vocabs.json", "r") as f:
            gloss_vocabs = json.load(f)
        token_to_gloss = {
            i: gloss for gloss, i in gloss_vocabs.items()
        }
    return token_to_gloss


def get_videos():
    global videos

    if videos is None:
        videos = []
        with open("data/CSL-Daily/sentence_label/csl2020ct_v2.pkl", "rb") as f:
            annos = pickle.load(f)
            for info in annos["info"]:
                videos.append({
                    "name": info["name"],
                    "gloss_sent": "".join(info["label_gloss"]),
                    "sentence": "".join(info["label_char"]),
                })
    return videos


def get_video_id(gloss_sent: str):
    videos = get_videos()

    for video in videos:
        if video["gloss_sent"] == gloss_sent:
            return video["name"]

    levenshtein = textdistance.Levenshtein()

    dists = []
    for video in videos:
        dists.append(levenshtein.normalized_similarity(video["gloss_sent"], gloss_sent))
    idx = np.argmax(dists)
    return videos[int(idx)]["name"]


def generate_video(name: str) -> Path:
    video_path = Path(f"generated_videos/{name}.mp4")
    if video_path.exists():
        return video_path

    image_paths = Path(f"data/CSL-Daily/frames_512x512/{name}/").glob("*.jpg")
    image_paths = sorted(image_paths)
    clip = ImageSequenceClip([str(image_path) for image_path in image_paths], fps=24)
    clip.write_videofile(str(video_path))

    return video_path

def translate(input_sentence: str):
    model, tokenizer = get_model_and_tokenizer()

    results = tokenizer(
        input_sentence,
        return_tensors="pt",
    )
    input_ids = results["input_ids"]

    with torch.no_grad():
        gloss_ids = model.generate(input_ids)[0].tolist()

    token_to_gloss = get_token_to_gloss()
    gloss_seq = [
        token_to_gloss[gloss_id] for gloss_id in gloss_ids if gloss_id in token_to_gloss
    ]
    video_name = get_video_id("".join(gloss_seq))
    # video_path = generate_video(video_name)

    print("input sentence:\t\t", input_sentence)
    print("translated sentence:\t", " ".join(gloss_seq))
    # print("video:\t", video_path)

    # return " ".join(gloss_seq), str(video_path.resolve())


def main():
    demo = gr.Interface(
        fn=translate,
        inputs="text",
        outputs=[
            "text",
            "video",
        ],
        allow_flagging=False,
        examples=[
            "你是老师吗？",
            "你是哪里人？",
            "我是上海人。",
            "中国菜很好吃。",
            "你会做中国菜吗？",
            "下午我想去超市。",
            "这个杯子十块钱。",
            "我儿子在医院工作，他是医生。",
        ]
    )

    demo.launch(share=True)


if __name__ == "__main__":
    main()
