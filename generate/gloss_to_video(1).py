import random
from pathlib import Path

from moviepy.editor import ImageSequenceClip


def main():
    gloss_seq = ["我", "爸爸", "是", "警察"]

    with open("data/CSL/dictionary.txt") as f:
        lines = f.readlines()
        lines = [line.strip().split("\t") for line in lines]
        gloss_to_id = {line[1]: int(line[0]) for line in lines}

    gloss_ids = [gloss_to_id[gloss] for gloss in gloss_seq if gloss in gloss_to_id]

    person_id = 1

    all_frame_paths = []

    for gloss_id in gloss_ids:
        frame_dir = Path(f"data/CSL/{gloss_id:03d}")
        frame_paths = [
            frame_path
            for frame_path in frame_dir.glob("P*")
            if frame_path.stem.startswith(f"P{person_id:02d}")
        ]
        frame_path = random.choice(frame_paths)
        # frame_path = sorted(frame_paths)[0]
        all_frame_paths += sorted(frame_path.glob("*.jpg"))

    clip = ImageSequenceClip([str(frame_path) for frame_path in all_frame_paths], fps=24)
    clip.write_videofile("generated_videos/v.mp4")


if __name__ == "__main__":
    main()
