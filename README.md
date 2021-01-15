# Language of Science
Extracting text from pdf files of journal articles from the Americal Journal of Sociology and the American Sociological Review (1896-2011) into .txt files. 

![pipeline](https://github.com/isaranwrap/Language-of-Science/blob/master/pipeline.png)
Naively, we would just pop the pdf into PyTesseract (Python wrapper for Google's Tesseract-OCR [Optical Character Recognition] Engine) and call it a day. But this only works if the pdfs/images you feed in are nice. And it turns out to be a little bit of a chicken-and-egg situation --> to find a good image text conder, you need to specify the crop ahead of time... but if you want a good crop, you need to know where your text is. So, how do we work arround this issue?

I start off by converting each page of the pdf into a folder containing the set of corresponding images - where each page is a separate image. I then turn each of those images into a np.uint8 (unsigned, 8-bit; i.e. maps between [0,255]) array. Once discretized, I run a Canny Edge Detection algorithm to trace out the text. After performing binary dilation on the images, the border surrounding the text becomes large enough that I can crop with OpenCV. Small complication(s) here: for the ASR Journal articles after 1946 and AJS Journal articles between 1946 and 1966, the format of the journal changed in that they switched to using double columns. The formatting is slightly different for different ASR articles too... some of the code in the pipeline is geared towards specific journals (as coarse-grained as possible). Using a lower number of binary dilation iterations (10 instead of 15) was necessary to ensure proper cropping for these journals. Otherwise, I ran through 15 "time-steps" of the dilation, if we think of dilating the image as running through time steps of the discretized heat equation. Then, instead of using the F1 metric to find the crop which maximizes the number of black pixels in the bounding box while minimizing area, I use OpenCV's findContour() function (which uses boundary tracing/border following) to trace out the contours of the boxes and rank order them by the area of the contour. Again, for the aforementioned articles where there were text separated into 2 columns, I picked the top 2 contours, and then compared their horizontal values to determine left versus right. Then, I [natural sorted](https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside) the image folder and used PyTesseract to covert the much more easily parseable images to text files. The PyTesseract output was run through a spell checker and the headers from a different pdf-to-txt conversion software (pdfminer.six) were added in. 

## How the files are arranged on my computer
I have a master folder with all of the AJS & ASR articles. Within the AJS pdf files, I split it into 3 time periods --> pre1946, 1946to1966, and post1971. Within the ASR pdf files, I simply split it into pre1946 and post1946. As long as the input "type" matches the subsection title, the code should run properly. 


## Additional resources

- [Automate the Boring Stuff](https://automatetheboringstuff.com/chapter13/)
- [GitHub OCR tutorial](https://github.com/wanghaisheng/awesome-ocr/wiki/Finding-blocks-of-text-in-an-image-using-Python,-OpenCV-and-numpy)
- [OpenCV finding contours](https://stackoverflow.com/questions/16538774/dealing-with-contours-and-bounding-rectangle-in-opencv-2-4-python-2-7)
- [OpenCV bounding rectangles](https://www.programcreek.com/python/example/89437/cv2.boundingRect)
