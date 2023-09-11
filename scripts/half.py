import cv2 as cv
from pathlib import Path

root = Path("/scratch/project_2008180/dataset")

files = root.rglob("*.tif")

for f in files:
    target = str(f).replace("dataset", "dataset_half")
    Path(target).parent.mkdir(parents=True, exist_ok=True)

    img = cv.imread(str(f), cv.IMREAD_GRAYSCALE)
    if "images" in target:
        img_half = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    else:
        img_half = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_NEAREST)

    cv.imwrite(target, img_half)
