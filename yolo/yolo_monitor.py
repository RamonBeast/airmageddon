import sys
import os
import torch
import cv2
from typing import Tuple
from pathlib import Path
from datetime import datetime
from utils.configuration import Configuration

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from yolo_utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from yolo_utils.general import (
    Profile,
    check_file,
    check_img_size,
    non_max_suppression,
    scale_boxes
)
from yolo_utils.torch_utils import select_device, smart_inference_mode

class YoloMonitor:
    def __init__(self, weights: str = None, source: str = None, 
                 imgsz: Tuple[int, int] = (1280, 768), conf_thres: float = 0.25, iou_thres: float = 0.45,
                 max_det: int = 100, device: str = "cpu", vid_stride: int = 1):
        self.conf = Configuration()
        self.weights = weights if weights != None else os.path.join(self.conf.get_config_param('models_folder'), self.conf.get_config_param('yolo_weights'))
        self.source = source if source != None else self.conf.get_config_param('camera_feed')
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.device = device
        self.vid_stride = vid_stride

        self.classes = None
        self.features = []
        self.is_warm = False
        self.dataset = None
        self.progressive = 1

    def warmup(self, data_source: str = None):
        if data_source is not None:
            self.source = data_source

        is_file = Path(self.source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
        is_url = self.source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
        self.webcam = self.source.isnumeric() or self.source.endswith(".streams") or (is_url and not is_file)
        screenshot = self.source.lower().startswith("screen")

        if is_url and is_file:
            self.source = check_file(self.source)  # download

        # Load model
        self.torch_device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.torch_device, dnn=False, data="", fp16=False)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size(self.imgsz, s=stride)  # check image size

        # Dataloader
        bs = 1  # batch_size

        if self.webcam:
            self.dataset = LoadStreams(self.source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)
            bs = len(self.dataset)
        elif screenshot:
            self.dataset = LoadScreenshots(self.source, img_size=imgsz, stride=stride, auto=pt)
        else:
            self.dataset = LoadImages(self.source, img_size=imgsz, stride=stride, auto=pt, vid_stride=self.vid_stride)

        self.iterator = iter(self.dataset)

        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup
        self.is_warm = True

    def get_similarity(self, feat_a, feat_b) -> float:
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6)

        # Image resolution has changed
        if len(feat_a) != len(feat_b):
            return -1.0

        return cos(feat_a, feat_b).tolist()

    def save_image(self) -> str:
        save_path = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_path += f'-{self.progressive:02d}.png'
        captures = os.path.join(self.conf.get_config_param('captures_folder'), save_path)

        if self.dataset.mode == "image":
            cv2.imwrite(captures, self.img)
        else:
            cv2.imwrite(captures, self.img[0])

        self.progressive += 1
        return save_path
    
    def get_path(self) -> str:
        return self.path
    
    def get_image(self):
        if self.dataset.mode == "image":
            img = self.img
        else:
            img = self.img[0] if isinstance(self.img, list) else self.img
            
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @smart_inference_mode()
    def run(self):
        if self.is_warm is False:
            self.warmup()

        seen, windows, dt = 0, [], (Profile(device=self.torch_device), Profile(device=self.torch_device), Profile(device=self.torch_device))

        path, im, im0s, vid_cap, s = next(self.iterator)
        self.img = im0s # if isinstance(im, Image.Image) else Image.fromarray(im)
        self.path = path

        #for path, im, im0s, vid_cap, s in self.dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            # Predict
            pred, features = self.model(im, augment=False, visualize=False)

            # Flatten
            features = features.view(-1)

            # Normalize
            features = torch.nn.functional.normalize(features, dim=0)

            # Save features
            self.features = features

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, False, max_det=self.max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if self.webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), self.dataset.count
                s += f"{i}: "
            else:
                p, im0, frame = path, im0s.copy(), getattr(self.dataset, "frame", 0)

            p = Path(p)  # to Path
            s += "%gx%g " % im.shape[2:]  # print string
            #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            self.detections = {}

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    self.detections[self.model.names[int(c)]] = n.tolist()

        return self.detections, self.features

            # Print time (inference-only)
            # LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / seen * 1e3 for x in dt)  # speeds per image
        # LOGGER.info(f"Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}" % t)

        # if update:
        #     strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)
