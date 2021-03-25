from pathlib import Path
import torch
from face_detector import FaceDetector
from mask_classifier import MaskClassifier
from mask_detector import MaskDetector

chckpts_p = Path("models", "trained_classifier", "checkpoints")
classifier_p = list(chckpts_p.glob("*.ckpt"))[-1]

classifier = MaskClassifier()
classifier.load_state_dict(torch.load(classifier_p), strict=False)

face_detector = FaceDetector(confidenceThreshold=0.3)

mask_detector = MaskDetector(face_detector, classifier, rate=4)

video_p = list(Path("data", "video").iterdir())[0]
output_p = Path("vid.mp4")

mask_detector.process_video(video_p, output_p, reduce_rate=3)
