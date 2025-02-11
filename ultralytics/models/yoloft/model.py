# Ultralytics YOLO 🚀, AGPL-3.0 license

from pathlib import Path

from ultralytics.engine.model import Model
from ultralytics.models import yoloft
from ultralytics.nn.tasks import  DetectionModel
from ultralytics.utils import ROOT, yaml_load


class YOLOFT(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolo11n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        path = Path(model)
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yoloft.detect.DetectionTrainer,
                "validator": yoloft.detect.DetectionValidator,
                "predictor": yoloft.detect.DetectionPredictor,
            }
        }
