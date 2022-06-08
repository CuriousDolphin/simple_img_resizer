import glob
import os
from pathlib import Path
import cv2
import xmltodict
import numpy as np

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    xyxy = np.copy(x)
    xyxy[0] = (x[0] + x[2]) / 2  # x center
    xyxy[1] = (x[1] + x[3]) / 2  # y center
    xyxy[2] = x[2] - x[0]  # width
    xyxy[3] = x[3] - x[1]  # height
    return xyxy


def img2label_paths(img_paths, label_path):
    # Define label paths as a function of image paths
    sa, sb = (
        os.sep + "images" + os.sep,
        os.sep + label_path + os.sep,
    )  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit(".", 1)[0] + ".xml" for x in img_paths]


def get_coco_annotations(annotations, image_id):
    res = []
    for i, annotation in enumerate(annotations):
        bboxwh = xyxy2xywh(
            [
                int(annotation["bndbox"]["xmin"]),
                int(annotation["bndbox"]["ymin"]),
                int(annotation["bndbox"]["xmax"]),
                int(annotation["bndbox"]["ymax"]),
            ]
        )
        res.append(
            {
                "id": i,
                "image_id": image_id,
                "category_id": annotations[i]["name"],  # TODO CONVERT xyxy
                "bbox": {
                    "x": int(bboxwh[0]),
                    "y": int(bboxwh[1]),
                    "width": int(bboxwh[2]),
                    "height": int(bboxwh[3]),
                },
            }
        )

    return res


def pascal_to_coco(label):
    image_id = label["annotation"]["filename"]

    if isinstance(label["annotation"]["object"], list):
        annotations = [ann for ann in label["annotation"]["object"]]
    else:
        annotations = [label["annotation"]["object"]]

    annotations = get_coco_annotations(annotations, image_id)

    coco_label = {
        "image": {
            "id": image_id,
            "filename": label["annotation"]["filename"],
            "width": int(label["annotation"]["size"]["width"]),
            "height": int(label["annotation"]["size"]["width"]),
        },
        "annotations": [
            ann for ann in annotations
        ],  # get_coco_annotations(label["annotation"]["object"], image_id),
    }
    return coco_label


class LoadImages:
    def __init__(self, path, labels_path):
        p = str(Path(path).resolve())  # os-agnostic absolute path
        if "*" in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, "*.*")))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f"ERROR: {p} does not exist")

        images = [x for x in files if x.split(".")[-1].lower() in IMG_FORMATS]
        ni = len(images)

        self.label_files = img2label_paths(images, labels_path)  # labels
        self.files = images
        self.nf = ni  # number of files

        assert self.nf > 0, (
            f"No images or videos found in {p}. "
            f"Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}"
        )

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]
        label_path = self.label_files[self.count]
        # Read image
        self.count += 1
        img0 = cv2.imread(path)  # BGR
        label = {}
        with open(label_path, "r") as xml_obj:
            label = xmltodict.parse(xml_obj.read())
            label = pascal_to_coco(label)
            xml_obj.close()

        assert img0 is not None, "Image Not Found " + path
        # print(f'image {self.count}/{self.nf} {path}: ', end='')
        return img0, path, label

    def __len__(self):
        return self.nf  # number of files

    def get_labels(self):
        return self.label_files
