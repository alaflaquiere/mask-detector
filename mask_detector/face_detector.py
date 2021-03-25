from pathlib import Path
import requests
import numpy as np
from cv2 import resize
from cv2.dnn import blobFromImage, readNetFromCaffe


URL_PROTOTXT = "https://raw.githubusercontent.com/JadHADDAD92/covid-mask-detector/master/covid-mask-detector/models/deploy.prototxt.txt"
URL_CAFFEMODEL = "https://raw.github.com/JadHADDAD92/covid-mask-detector/master/covid-mask-detector/models/res10_300x300_ssd_iter_140000.caffemodel"


class FaceDetector:
    """ Face Detector class
    """

    def __init__(self,
                 model_p: Path = Path("models", "face_detector"),
                 download=True,
                 confidenceThreshold: float = 0.6):
        self.model_p = model_p
        self.confidenceThreshold = confidenceThreshold
        if download:
            self._download()
        else:
            self._check_files()
        self.classifier = self._get_classifier()

    def _download(self):
        self.model_p.mkdir(parents=True, exist_ok=True)
        r = requests.get(URL_PROTOTXT)
        (self.model_p / "deploy.prototxt.txt").write_text(r.text)
        r = requests.get(URL_CAFFEMODEL)
        (self.model_p / "res10_300x300_ssd_iter_140000.caffemodel").write_bytes(r.content)

    def _check_files(self):
        prototxt_p = self.model_p / "deploy.prototxt.txt"
        caffemodel_p = self.model_p / "res10_300x300_ssd_iter_140000.caffemodel"
        if not prototxt_p.exists():
            print("{} file missing".format(prototxt_p))
        if not caffemodel_p.exists():
            print("{} file missing".format(caffemodel_p))

    def _get_classifier(self):
        prototxt_p = self.model_p / "deploy.prototxt.txt"
        caffemodel_p = self.model_p / "res10_300x300_ssd_iter_140000.caffemodel"
        return readNetFromCaffe(str(prototxt_p), str(caffemodel_p))

    def detect(self, image):
        """ detect faces in image
        """
        net = self.classifier
        height, width = image.shape[:2]
        blob = blobFromImage(resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()
        faces = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidenceThreshold:
                continue
            box = detections[0, 0, i, 3:7] * \
                np.array([width, height, width, height])
            startX, startY, endX, endY = box.astype("int")
            faces.append(np.array([startX, startY, endX-startX, endY-startY]))
        return faces
