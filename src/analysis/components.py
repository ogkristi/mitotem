from typing import Callable
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import easyocr

def detect_scale(img: np.ndarray, reader):
    hist = cv.calcHist([img], channels=[0], mask=None, histSize=[256], ranges=[0,256]).ravel()
    
    if hist[0] > hist[255]: # make sure scalebar is white
        img = cv.bitwise_not(img)
    _, bin = cv.threshold(img,254,255,cv.THRESH_BINARY)

    kwidth = bin.shape[1]//8
    # use wide structuring element to erase everything but scalebar
    scalebar = open_rc(bin, cv.getStructuringElement(cv.MORPH_RECT,(kwidth,1)))
    contours, _ = cv.findContours(scalebar, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    bar = contours[0]
    length = bar[:,:,0].max()-bar[:,:,0].min()+1 # scalebar length in pixels

    # use wide structuring element to merge text as single object
    dil = cv.dilate(bin, cv.getStructuringElement(cv.MORPH_RECT,(kwidth//2,1)))
    contours, _ = cv.findContours(dil, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    ctr_biggest = max(contours, key=cv.contourArea)
    x, y, w, h = cv.boundingRect(ctr_biggest)
    roi = bin[y:y+h,x:x+w] # rough crop of text
    roi = cv.resize(roi, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
    
    value, unit, *_ = reader.readtext(roi, detail=0) # character recognition

    if value.isnumeric():
        if unit == 'um':
            return float(value)/length, 'micron'
        elif unit == 'nm':
            return 0.001*float(value)/length, 'micron'

    return 1., 'pixel'

def get_cristae_mask(mito: np.ndarray, mask: np.ndarray, operations: list[Callable[..., np.ndarray]]):
    img = np.copy(mito)

    for op in operations:
        img = op(img)
    
    return img

def bottomhat(src, ksize, type):
    se = cv.getStructuringElement(type,(ksize,ksize))
    dst = cv.morphologyEx(src, op=cv.MORPH_BLACKHAT, kernel=se)
    
    return cv.bitwise_not(dst)

def bottomhat_rc(src, ksize, type):
    dst = close_rc(src, ksize, type) - src

    return cv.bitwise_not(dst)

def area_filter(img: np.ndarray, threshold: int):
    cc = cv.connectedComponentsWithStats(img, connectivity=8, ltype=cv.CV_32S)
    (n, labels, stats, centroids) = cc
    stats_df = pd.DataFrame(stats, columns=['x','y','w','h','area'])
    small = stats_df[stats_df['area'] < threshold].index.to_numpy()

    for i in small:
        img[labels == i] = 0

    return img

def open_rc(img: np.ndarray, kernel: np.ndarray):
    binary = True if len(np.unique(img)) == 2 else False
    mask = np.copy(img)

    marker = cv.erode(img, kernel=kernel)
    marker_old = None

    se = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    while not np.array_equal(marker, marker_old):
        marker_old = np.copy(marker)
        marker = cv.dilate(marker,kernel=se)
        if binary:
            marker = cv.bitwise_and(marker,mask)
        else:
            marker = np.minimum(marker,mask)

    return marker

def close_rc(img: np.ndarray, ksize: int, type):
    binary = True if len(np.unique(img)) == 2 else False
    mask = np.copy(img)

    kernel = cv.getStructuringElement(type,(ksize,ksize))
    marker = cv.dilate(img, kernel=kernel)
    marker_old = None

    se = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    while not np.array_equal(marker, marker_old):
        marker_old = np.copy(marker)
        marker = cv.erode(marker,kernel=se)
        if binary:
            marker = cv.bitwise_or(marker,mask)
        else:
            marker = np.maximum(marker,mask)

    return marker

def dilate_rc(marker: np.ndarray, mask: np.ndarray):
    binary = True if len(np.unique(marker)) == 2 else False

    marker_old = None
    se = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    while not np.array_equal(marker, marker_old):
        marker_old = np.copy(marker)
        marker = cv.dilate(marker,kernel=se)
        if binary:
            marker = cv.bitwise_and(marker,mask)
        else:
            marker = np.minimum(marker,mask)

    return marker

def close(img: np.ndarray, ksize: int):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(ksize,ksize))
    return cv.morphologyEx(img, op=cv.MORPH_CLOSE, kernel=kernel)

def open(img: np.ndarray, ksize: int):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(ksize,ksize))
    return cv.morphologyEx(img, op=cv.MORPH_OPEN, kernel=kernel)

def dilate(img: np.ndarray):
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    return cv.dilate(img, kernel=kernel)

def prune(img: np.ndarray, n: int):
    endpoint1 = np.array([[0,-1,-1,],[1,1,-1],[0,-1,-1]], dtype="int")
    endpoint2 = np.array([[1,-1,-1,],[-1,1,-1],[-1,-1,-1]], dtype="int")
    B = []
    for _ in range(4):
        B.append(endpoint1)
        B.append(endpoint2)
        endpoint1 = np.rot90(endpoint1)
        endpoint2 = np.rot90(endpoint2)

    # thinning
    thinned = np.copy(img)
    for _ in range(n):
        ends = np.zeros_like(img)
        for kernel in B:
            ends += cv.morphologyEx(thinned, op=cv.MORPH_HITMISS, kernel=kernel)
        thinned -= ends

    # find endpoints
    ends = np.zeros_like(img)
    for kernel in B:
        ends += cv.morphologyEx(thinned, op=cv.MORPH_HITMISS, kernel=kernel)

    # dilate endpoints n times using original img as delimiter
    for _ in range(n):
        ends = cv.dilate(ends, kernel=np.ones((3,3), np.uint8))
        ends = np.bitwise_and(ends, img)
    
    return np.bitwise_or(ends, thinned)

def holefill(img: np.ndarray):
    binary = True if len(np.unique(img)) == 2 else False
    
    if binary:
        mask = cv.bitwise_not(img)
        marker = np.copy(mask)
        marker[1:-1,1:-1] = 0
    else:
        mask = img
        marker = np.copy(img)
        marker[1:-1,1:-1] = np.max(marker)

    marker_old = None
    se = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    while not np.array_equal(marker, marker_old):
        marker_old = np.copy(marker)
        if binary:
            marker = cv.dilate(marker,kernel=se)
            marker = cv.bitwise_and(marker,mask)
        else:
            marker = cv.erode(marker,kernel=se)
            marker = np.maximum(marker,mask)

    return cv.bitwise_not(marker) if binary else marker

def contour_smoothing(src: np.ndarray, p: float):
    contours, _ = cv.findContours(src, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    dst = np.zeros_like(src)

    q = 1.-p
    for ctr in contours:
        l = np.int32(0.5*q*len(ctr))
        fd = np.fft.fftshift(cv.dft(np.float32(ctr), flags=cv.DFT_COMPLEX_INPUT+cv.DFT_COMPLEX_OUTPUT))
        if l != 0:
            fd[:l] = [[0.,0.]]
            fd[-l:] = [[0.,0.]]
        newctr = np.int32(cv.idft(np.fft.ifftshift(fd), flags=cv.DFT_COMPLEX_OUTPUT+cv.DFT_SCALE))
        dst = cv.drawContours(dst, [newctr.astype("int")], 0, 255, cv.FILLED)

    return dst

def demaze(img: np.ndarray) -> np.ndarray:
    """Removes maze artifacts by filtering input image in the frequency domain
    using four gaussian shaped notch filters located midway at image borders. 

    Args:
        img (np.ndarray): Single-channel image.

    Returns:
        np.ndarray: De-mazed image.
    """
    X = cv.dft(np.float32(img),flags=cv.DFT_COMPLEX_OUTPUT)
    X = np.fft.fftshift(X)

    m, n, _ = X.shape
    x_coords = np.arange(-n//2,n//2)
    y_coords = np.arange(-m//2,m//2)
    xv, yv = np.meshgrid(x_coords,y_coords)

    cutoff_edge = 0.6
    cutoff_center = 0.3
    a_v = int(cutoff_edge*n/2)
    b_v = int(cutoff_center*m/2)
    a_h = int(cutoff_center*n/2)
    b_h = int(cutoff_edge*m/2)

    cx, cy = n//2, m//2 # centers of four notch filters (0,-cy),(-cx,0),(cx,0),(0,cy)
    H = (np.exp(-((xv/a_v)**2 + ((yv-cy)/b_v)**2)/2)
       + np.exp(-(((xv-cx)/a_h)**2 + (yv/b_h)**2)/2)
       + np.exp(-(((xv+cx)/a_h)**2 + (yv/b_h)**2)/2)
       + np.exp(-((xv/a_v)**2 + ((yv+cy)/b_v)**2)/2))
    H = 1.-H
    Y = X*H[:,:,None]

    y = cv.idft(np.fft.ifftshift(Y), flags=cv.DFT_REAL_OUTPUT+cv.DFT_SCALE)
    y = cv.equalizeHist(cv.normalize(y, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U))

    return y

def get_mito_stats(img: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    mask = cv.normalize(mask, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    stats = []
    for cnt in contours:
        M = cv.moments(cnt)
        mitostats = {
            'centroid': (int(M['m10']/M['m00']), int(M['m01']/M['m00'])), # xy
            'area': M['m00'],
            'perimeter': cv.arcLength(cnt,True),
            'bbox': cv.boundingRect(cnt) # xywh
            }
        
        x,y,w,h = mitostats['bbox']
        imask = np.zeros_like(img) # instance mask
        imask = cv.drawContours(imask, [cnt], 0, 255, cv.FILLED)
        instance = cv.bitwise_and(img, imask)
        instance = instance[y:y+h, x:x+w]
        
        #get_cristae_stats(instance)
        stats.append(mitostats)

    return pd.DataFrame(stats)

def crop_all_mitos(img: np.ndarray, mask: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    mask = cv.normalize(mask, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    mitos, imasks = [], []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        mitos.append(img[y:y+h, x:x+w])

        imask = np.zeros((h,w), dtype=np.uint8) # instance mask
        cnt_shifted = cnt-np.array([[x,y]])
        imask = cv.drawContours(imask, [cnt_shifted], 0, 255, cv.FILLED)
        imasks.append(imask)

    return (mitos, imasks)