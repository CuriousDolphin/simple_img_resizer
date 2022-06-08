import sys
import os
import click
import cv2
from utils.dataloader import LoadImages
import json

MAX_HEIGHT = 450
MAX_WIDTH = 800


def update_annotations(annotations, width_ratio, height_ratio):
    new_annotations = []
    for ann in annotations:
        x = ann["bbox"]["x"]
        y = ann["bbox"]["y"]
        width = ann["bbox"]["width"]
        height = ann["bbox"]["height"]
        ann["bbox"]["x"] = int(x * width_ratio)
        ann["bbox"]["y"] = int(y * height_ratio)
        ann["bbox"]["width"] = int(width * width_ratio)
        ann["bbox"]["height"] = int(height * height_ratio)
        new_annotations.append(ann)

    return new_annotations


class Resizer:
    def __init__(self, imagedir, xmldir, outputdir):
        self.dataloader = LoadImages(imagedir, xmldir)
        self.outputdir = outputdir

    def resize(self):
        annotations = []
        images = []
        for i, (im, path, label) in enumerate(self.dataloader):
            print(f"Current image {path} {im.shape}")
            image_label = label["image"]
            image_annotations = label["annotations"]
            print(f"Found {len(image_annotations)} annotations")
            height = im.shape[0]
            width = im.shape[1]
            need_resize = height > MAX_HEIGHT or width > MAX_WIDTH
            if need_resize:  # Resize
                new_height = MAX_HEIGHT if height > MAX_HEIGHT else height
                new_width = MAX_WIDTH if width > MAX_WIDTH else width
                im = cv2.resize(
                    im, (new_width, new_height)
                )  # TODO better resize with aspect ratio
                print(f"Need Resize, New shape {im.shape}")
                image_label["width"] = new_width
                image_label["height"] = new_height  #
                image_annotations = update_annotations(
                    image_annotations, new_width / width, new_height / height
                )
            images.append(image_label)
            annotations.extend(image_annotations)
            img_new_path = f'{self.outputdir}/images/{image_label["id"]}'
            cv2.imwrite(img_new_path, im)
            print(f"Saved image at : {img_new_path} \n")
            # cv2.imshow("image", im)
            # cv2.waitKey(1000)

        coco_annotations = {
            "categories": ["COCO_CATEGORIES"],
            "images": images,
            "annotations": annotations,
        }
        ann_path = f"{self.outputdir}/coco.json"
        with open(ann_path, "w") as fp:
            json.dump(dict(coco_annotations), fp, indent=4)

        print(f"Saved COCO annotations at : {ann_path}")


@click.command()
@click.option("--imagedir")
@click.option("--xmldir")
@click.option("--outputdir")
def main(imagedir, xmldir, outputdir):
    resizer = Resizer(imagedir, xmldir, outputdir)
    resizer.resize()


if __name__ == "__main__":
    main()
