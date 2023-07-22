import cv2 as cv
import numpy as np
import pandas as pd

def get_mito_stats(mask: np.ndarray) -> pd.DataFrame:
    mask = cv.normalize(mask, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    stats = []
    for cnt in contours:
        M = cv.moments(cnt)
        stats.append({
            'centroid': (int(M['m10']/M['m00']), int(M['m01']/M['m00'])), # xy
            'area': M['m00'],
            'perimeter': cv.arcLength(cnt,True),
            'bbox': cv.boundingRect(cnt) # xywh
            })

    return pd.DataFrame(stats)