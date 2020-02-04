import os
os.chdir('/home/ishi/Desktop/fall2019/socResearch')
from pdf2text_utils import *

baseFolder = '/home/ishi/Desktop/fall2019/socResearch'

## For AJS - format briefly changed from 1946-1966... for ASR format changed for good in 1946

# AJS, pre1946
fileFolder = os.path.join(baseFolder, 'Language of science - AJS-ASR/AJS pdf files/pre1946')
for file in os.listdir(fileFolder)[2:]:
    imageFolder = pdftoCroppedImages(os.path.join(fileFolder, file), type='AJS_pre1946')
    convertImagestoStrings(imageFolder)
print('AJS, pre1946 done!')
# AJS, post1971
fileFolder = os.path.join(baseFolder, 'Language of science - AJS-ASR/AJS pdf files/post1971')
for file in os.listdir(fileFolder)[2:]:
    imageFolder = pdftoCroppedImages(os.path.join(fileFolder, file), type='AJS_post1971')
    convertImagestoStrings(imageFolder)
print('AJS, post1971 done!')
#ASR, pre1946
fileFolder = os.path.join(baseFolder, 'Language of science - AJS-ASR/ASR pdf files/pre1946')
for file in os.listdir(fileFolder)[2:]:
    imageFolder = pdftoCroppedImages(os.path.join(fileFolder, file), type='ASR_pre1946')
    convertImagestoStrings(imageFolder)
print('ASR, pre1946 done!')
## Now, we're dealing with the double column format change... 
    
#AJS, 1946to1966
fileFolder = os.path.join(baseFolder, 'Language of science - AJS-ASR/AJS pdf files/1946to1966')
for file in os.listdir(fileFolder)[2:]:
    imageFolder = pdftoCroppedImages(os.path.join(fileFolder, file), type='AJS_1946to1966')
    convertImagestoStrings(imageFolder)
print('AJS, 1946to1966 done!')
    
#ASR, post1946
fileFolder = os.path.join(baseFolder, 'Language of science - AJS-ASR/ASR pdf files/post1946')
for file in os.listdir(fileFolder)[2:]:
    imageFolder = pdftoCroppedImages(os.path.join(fileFolder, file), type='ASR_post1946')
    convertImagestoStrings(imageFolder)
print('ASR, post1946 done!')
print('All done, take a look!')  
