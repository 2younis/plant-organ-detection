import os
import torch
from pathlib import Path
from detectron2.data import MetadataCatalog
from utils.annotations_io import Writer


cpu_device = torch.device("cpu")

def annotator(cfg, args, predictions, image, path, out_dir):

    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
    )

    folder = os.path.normpath(args.output)

    class_names = metadata.get("thing_classes", None)
    image = image[:, :, ::-1]
    image_id = Path(path).stem
    annotation = Writer(folder.split(os.sep)[-1], os.path.basename(path),
                        list(image.shape), False, path)
    instances = predictions["instances"].to(cpu_device)
    boxes = instances.pred_boxes  # if predictions.has("pred_boxes") else None
    classes = instances.pred_classes
    labels = [class_names[i] for i in classes]
    boxes = boxes.tensor.numpy()
    num_instances = len(boxes)
    scores = instances.scores
    scores = scores.numpy()

    for i in range(num_instances):
        xmin, ymin, xmax, ymax = boxes[i]
        annotation.addBndBox(int(xmin), int(ymin), int(
            xmax), int(ymax), labels[i], scores[i])

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    annotation.save(out_dir + image_id + '.xml')
