import numpy as np

import cv2 as cv
import os 
import re
import scipy as sci
import scipy.ndimage

import pytesseract
from pdf2image import convert_from_path
from PIL import Image

def binary_dilation(arr, N=5):
    bd = sci.ndimage.morphology.binary_dilation(arr)
    for _ in range(N-1):
        bd = sci.ndimage.morphology.binary_dilation(bd)
    return bd

def pdftoCroppedImages(filePath, baseFolder = '/home/ishi/Desktop/fall2019/socResearch/', type = None):
    fileName = os.path.split(filePath)[1]
    if type == 'AJS_pre1946':
        folderName = str(baseFolder + 'Language of Science Images/AJS_pre1946/' + fileName)[:-4] 
    elif type == 'AJS_1946to1966':
        folderName = str(baseFolder + 'Language of Science Images/AJS_1946to1966/' + fileName)[:-4]
    elif type == 'AJS_post1971':
        folderName = str(baseFolder + 'Language of Science Images/AJS_post1971/' + fileName)[:-4]
     
    elif type == 'ASR_pre1946':
        folderName = str(baseFolder + 'Language of Science Images/ASR_pre1946/' + fileName)[:-4]
    elif type == 'ASR_post1946':
        folderName = str(baseFolder + 'Language of Science Images/ASR_post1946/'+ fileName)[:-4]
    os.mkdir(folderName)
    pages = convert_from_path(filePath)
    for indx, i in enumerate(pages):
        img = np.uint8(i)[:,:,0]
        edges = cv.Canny(img, 5, 5)
        
        if type == 'AJS_1946to1966' or type == 'ASR_post1946':
            dilated_img = np.uint8(binary_dilation(edges, 10)) # Use 10 instead of 15 for num_dilations
        else:
            dilated_img = np.uint8(binary_dilation(edges, 15))

        ret, thresh = cv.threshold(dilated_img, 0.5, 1, 0)
        #contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cnts = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        cnts = sorted(cnts, key=cv.contourArea, reverse=True)
        
        if type == 'AJS_pre1946' or type == 'AJS_post1971' or type=='ASR_pre1946':
            for c in cnts:
                x,y,w,h = cv.boundingRect(c)
                ROI = img[y:y+h, x:x+w]
                break
            
            out_im = Image.fromarray(ROI)
            out_im.save('{}/pg{}.png'.format(folderName, indx+1))
        elif type == 'AJS_1946to1966' or type == 'ASR_post1946':
            for c in cnts[:1]:
                x1, y1, w1, h1 = cv.boundingRect(c)
            for c in cnts[:2]:
                x2, y2, w2, h2 = cv.boundingRect(c)
            
            if x1 < x2:
                ROI_L = img[y1:y1+h1, x1:x1+w1]
                ROI_R = img[y2:y2+h2, x2:x2+w2]
            elif x1 > x2:
                ROI_R = img[y1:y1+h1, x1:x1+w1]
                ROI_L = img[y2:y2+h2, x2:x2+w2]
            
            imout_R = Image.fromarray(ROI_R)
            imout_L = Image.fromarray(ROI_L)
            imout_L.save('{}/pg{}_1.png'.format(folderName, indx+1))
            imout_R.save('{}/pg{}_2.png'.format(folderName, indx+1))
    return '{}/'.format(folderName)

def getInt(text):
    return int(text) if text.isdigit() else text

#https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
def natural_sort(text):
    return [ getInt(c) for c in re.split(r'(\d+)', text) ]


def convertImagestoStrings(imageFolder):
    images_list = os.listdir(imageFolder)
    images_list.sort(key=natural_sort)
    
    txt_list = list()
    for imgFile in images_list:
        file = imageFolder + imgFile
        img = Image.open(file)
        txt = pytesseract.image_to_string(img)
        
        #Do we want to replace new lines with spaces?
        #txt.replace('\n', ' ')
        txt_list.append(txt)
    
    with open(os.path.join(imageFolder,'out.txt'), 'w') as f:
        for page in txt_list:
            f.write("{}\n".format(page))
