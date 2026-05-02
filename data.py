import os
import cv2
import torch
import numpy as np
import pandas as pd
from typing import Iterator

class DataLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_dir = os.path.join(root_dir, "ydata-tvsum50-data", "data")
        self.video_dir = os.path.join(root_dir, "ydata-tvsum50-video", "video")

        self.anno_file = os.path.join(self.data_dir, "ydata-tvsum50-anno.tsv")
        self.info_file = os.path.join(self.data_dir, "ydata-tvsum50-info.tsv")

        self.info_df = pd.read_csv(self.info_file, sep="\t")
        self.anno_df = pd.read_csv(self.anno_file, sep="\t", names=['video_id', 'category', 'importance_scores'])

        self.video_names = self.info_df["video_id"].tolist()
        self.annotations = self._parse_annotations()

    def _parse_annotations(self):
        annotations = {}

        for video_id in self.video_names:
            video_rows = self.anno_df[self.anno_df["video_id"] == video_id]

            # Each row corresponds to one user
            user_scores = []
            for _, row in video_rows.iterrows():
                scores = np.array(
                    list(map(int, row["importance_scores"].split(","))),
                    dtype=np.float32
                )
                user_scores.append(scores)

            annotations[video_id] = np.stack(user_scores, axis=0)

        return annotations

    def get_video_annotation(self, video_id):
        return self.annotations[video_id]

    def get_video_frames(self, video_id: str, batch_size=32) -> Iterator[torch.Tensor]:
        video_path = os.path.join(self.video_dir, f"{video_id}.mp4")
        cap = cv2.VideoCapture(video_path)

        finished = False
        try:
            while not finished:
                frames = []

                for i in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        finished = True
                        break

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # BGR → RGB
                    frame = torch.from_numpy(frame).permute(2, 0, 1) # HWC → CHW
                    frames.append(frame)

                if len(frames) > 0:
                    yield torch.stack(frames)
        finally:
            cap.release()
