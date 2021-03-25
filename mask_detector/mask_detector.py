from pathlib import Path
import numpy as np
import cv2
import torch
from tqdm import tqdm
from skvideo.io import FFmpegWriter, vreader, ffprobe
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor

from face_detector import FaceDetector
from mask_classifier import MaskClassifier


class MaskDetector():

    def __init__(self, face_detector: FaceDetector,
                 mask_classifier: MaskClassifier,
                 rate: int = 4):
        self.face_detector = face_detector
        self.mask_classifier = mask_classifier
        self.mask_classifier.eval()
        self.transformations = Compose([
            ToPILImage(),
            Resize((100, 100)),
            ToTensor(),
        ])
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.labels = ['No mask', 'Mask']
        self.labelColor = [(10, 0, 255), (10, 255, 0)]

    def process_frame(self, frame: np.ndarray):
        faces = self.face_detector.detect(frame)
        for face in faces:
            xStart, yStart, width, height = face

            # clamp coordinates that are outside of the image
            xStart, yStart = max(xStart, 0), max(yStart, 0)

            # predict mask label on extracted face
            faceImg = frame[yStart:yStart+height, xStart:xStart+width]
            faceImg = self.transformations(faceImg).unsqueeze(0)
            with torch.no_grad():
                output = self.mask_classifier(
                    faceImg.to(self.mask_classifier.device))
            _, predicted = torch.max(output.data, 1)

            # draw face frame
            cv2.rectangle(frame,
                          (xStart, yStart),
                          (xStart + width, yStart + height),
                          (126, 65, 64),
                          thickness=2)

            # center text according to the face frame
            textSize = cv2.getTextSize(
                self.labels[predicted], self.font, 1, 2)[0]
            textX = xStart + width // 2 - textSize[0] // 2

            # draw prediction label
            cv2.putText(frame,
                        self.labels[predicted],
                        (textX, yStart-20),
                        self.font, 1, self.labelColor[predicted], 2)
        return frame

    def process_video(self, video_p: Path, output_p: Path, reduce_rate: int = 1):
        meta = ffprobe(video_p)
        nb_frames = int(meta["video"]["@nb_frames"])
        frames = vreader(str(video_p))
        writer = FFmpegWriter(str(output_p),
                              outputdict={"-r": str(int(30/reduce_rate))}
                              )
        for i, frame in enumerate(tqdm(frames, total=nb_frames)):
            if i % reduce_rate == 0:
                frame = self.process_frame(frame)
                writer.writeFrame(frame)
        writer.close()
