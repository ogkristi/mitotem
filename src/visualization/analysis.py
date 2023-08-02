import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def show_mito_stats(img: np.ndarray, stats: pd.DataFrame) -> None:
    box_color = (255,255,0)
    box_thickness = 5
    font = cv.FONT_HERSHEY_SIMPLEX
    text_color = (0,0,0)
    text_scale = 1.2
    text_thickness = 1

    img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    labelmask = np.zeros(img.shape[:-1], dtype=np.uint8)
    for _, row in stats.iterrows():
        x, y, w, h = row.bbox
        cv.rectangle(img,(x,y),(x+w,y+h),box_color,box_thickness)

        label = f"{row.centroid[0]},{row.centroid[1]}"
        (w, h), baseline = cv.getTextSize(label, font, text_scale, text_thickness)

        # Draw label in a tight box in the upper left corner of bounding box
        cv.rectangle(img, (x, y-box_thickness//2), (x+w, y+h+baseline), box_color, -1)
        cv.putText(labelmask, label, (x, y+h), font, text_scale, 255, text_thickness)
    
    img[labelmask != 0] = text_color
    plt.figure(figsize=(16,10))
    plt.imshow(img)