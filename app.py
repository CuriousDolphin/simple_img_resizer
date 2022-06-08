import sys
import os
import click
import cv2
from utils.dataloader import LoadImages
import json

MAX_HEIGHT = 450
MAX_WIDTH = 800


def resize_img(image, x, y):
    return cv2.resize(image, (x, y), interpolation=cv2.INTER_LANCZOS4)


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

            height = im.shape[0]
            width = im.shape[1]
            if height > MAX_HEIGHT or width > MAX_WIDTH:  # Resize
                new_height = MAX_HEIGHT if height > MAX_HEIGHT else height
                new_width = MAX_WIDTH if width > MAX_WIDTH else width
                im = cv2.resize(
                    im, (new_width, new_height)
                )  # TODO better resize with aspect ratio
                print(f"Need Resize, New shape {im.shape}")
                image_label["width"] = new_width
                image_label["height"] = new_height  #
                # TODO resize annotations, no time :(
            images.append(image_label)
            annotations.extend(image_annotations)
            cv2.imwrite(f'{self.outputdir}/images/{image_label["id"]}', im)
            print("\n")
            # cv2.imshow("image", im)
            # cv2.waitKey(1000)

        coco_annotations = {
            "categories": ["COCO_CATEGORIES"],
            "images": images,
            "annotations": annotations,
        }

        with open(f"{self.outputdir}/coco.json", "w") as fp:
            json.dump(dict(coco_annotations), fp, indent=4)


@click.command()
@click.option("--imagedir")
@click.option("--xmldir")
@click.option("--outputdir")
def main(imagedir, xmldir, outputdir):
    resizer = Resizer(imagedir, xmldir, outputdir)
    resizer.resize()


if __name__ == "__main__":
    main()
