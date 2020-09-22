#!/usr/bin/env python
# coding: utf-8

# ### Imports, Reading in files & Defining functions

# In[1]:


import os
import io
import regex as re
import numpy as np
from tqdm import tqdm
from termcolor import colored
from colorama import Back, Style

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = [word.upper() for word in stopwords.words('english')]
hard_coded_non_section_words = ['URL', 'UTC', 'JSTOR', 'AMERICAN', 'JOURNAL', 'SOCIOLOGY', 'ABSTRACT', 'TABLE', 'CHART',
                                'AMERICAN JOURNAL OF SOCIOLOGY', 'UNIVERSITY OF CHICAGO', 'AMERICA', 'JOURNAL OF SOCIOLOGY',
                                'REFERENCES', 'AMERICAN SOCIOLOGICAL REVIEW', 'AMERICAN SOCIOLOGICAL ASSOCIATION']

def PDFtoString(filePath, pdfFolder=None):
    
    out = io.StringIO()
    if pdfFolder is not None:
        filePath = os.path.join(pdfFolder, filePath)
    with open(filePath, 'rb') as f:
        parser = PDFParser(f)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, out, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for indx, page in enumerate(PDFPage.create_pages(doc)):
            #if indx != 0:
            interpreter.process_page(page)
            
    return out.getvalue() 

def printMetaInfo(convertedStrings, pdfPaths, journal = 'AJS'):
    
    # Print the regex matches for access date, abstract, table/chart, and references
    for indx, string in enumerate(convertedStrings):
        print(indx, colored(pdfPaths[indx], 'red'))
        if re.findall(r'Accessed: \d{2}-\d{2}-\d{4} \d{2}:\d{2} UTC', string):
            print('\t', colored(re.findall(r'Accessed: \d{2}-\d{2}-\d{4} \d{2}:\d{2} UTC', string)[0], 'blue'))
        print('\t', colored('Abstract match:', 'magenta'), colored(re.search(r'ABSTRACT', string), 'magenta'))
        
        # Chart, table, or figure matches
        matches = re.finditer(r'CHART', string)
        for match in matches:
            print('\t', colored('Chart match:', 'green'), colored(match, 'green'))
        matches = re.finditer(r'TABLE', string)
        for match in matches:
            print('\t', colored('Table match:', 'green'), colored(match, 'green'))
        matches = re.finditer(r'FIG|FIGURE', string)
        for match in matches:
            print('\t', colored('Figure match:', 'green'), colored(match, 'green'))  
            
        print('\t', colored('Reference match:', 'magenta'), colored(re.search(r'REFERENCES', string), 'magenta'))
        
        # This finds the title words which aren't stop words & aren't digits and upper cases them all
        if journal == 'AJS':
            titles = [re.findall(r'(?<=AJS_\d{4}_\d{1,3}_\d{1,2}_).*(?=.pdf)', file_name)[0] for file_name in pdfPaths]
        elif journal == 'ASR':
            titles = [re.findall(r'(?<=ASR_\d{4}_\d{1,3}_\d{1,2}_).*(?=.pdf)', file_name)[0] for file_name in pdfPaths]
        title_words = [i for i in titles[indx].upper().split() if i not in stop_words and not i.isdigit()]
        
        # Then adds them to stop words & other hard-coded regularly occuring words
        non_section_words = hard_coded_non_section_words + title_words + stop_words 
        non_section_words += [title.upper() for title in titles] + [title.strip('The ').upper() for title in titles]
        
        matches = re.finditer(r'[A-Z]{4,}(\s+?[A-Z]{2,}){0,}', string)
        sections = dict()
        
        for match in matches:
            if match.group() not in non_section_words and match not in sections:
                sections[match.group()] = match.span()
        for section in sections:
            print('\t', colored('Section match:', 'blue'), colored([section, sections[section]], 'blue'))

def createOutputStrings(convertedStrings, folder = None, journal = 'AJS'):
    out = list() #Output strings go here, hopefully one out per pdf s.t. len(out) == len(convertedStrings)
    for indx, string in enumerate(convertedStrings):
        # Define meta-information
        outString = '-------------\n||Meta-info||\n-------------\n'

        # The second condition is more restrictive, so check that that's not None.
        header = article = None
        if re.search(r'extend access to\n(.)+?(?=C)(.)+?(?=\w{2,})', string) is not None:
            header = string[:re.search(r'extend access to\n(.)+?(?=C)', string).end()].strip()
            if re.search(r' Your use of the JSTOR', header) is not None:
                header = header[:re.search(r' Your use of the JSTOR', header).span()[0]]
            article = string[re.search(r'extend access to\n(.)+?(?=C)(.)+?(?=\w{2,})', string).end():]
            outString += re.sub(r'Accessed: \d{2}-\d{2}-\d{4} \d{2}:\d{2} UTC', '', header)

        # Add access date
        outString += '\n\n' + re.findall(r'Accessed: \d{2}-\d{2}-\d{4} \d{2}:\d{2} UTC', string)[0] + '\n'

        # Add abstract info
        abstract = re.search(r'ABSTRACT', string)
        if abstract is not None:
            outString += 'Abstract match at {}'.format(abstract.span()) + '\n'
        elif abstract is None:
            outString += 'No abstract found.\n'

        # Add chart info
        chart_matches = re.finditer(r'CHART', string)    
        if chart_matches is not None:
            for match in chart_matches:
                outString += 'Chart match at {}'.format(match.span()) + '\n'

        # Add table info 
        table_matches = re.finditer(r'TABLE', string)
        if table_matches is not None:
            for match in table_matches:
                outString += 'Chart match at {}'.format(match.span()) + '\n'
                
        # Add figure info
        figure_matches = re.finditer(r'FIG|FIGURE', string)
        if figure_matches is not None:
            for match in figure_matches:
                outString += 'Figure match at {}'.format(match.span()) + '\n'

        # Add references info
        references = re.search(r'REFERENCES', string)
        if references is not None:
            outString += 'Reference match at {}'.format(references.span()) + '\n'
        elif references is None:
            outString += 'No reference section found.\n'

        # Add section info 

        ### First, look for the title words and tokenize them.
        if journal == 'AJS':
            titles = [re.findall(r'(?<=AJS_\d{4}_\d{1,3}_\d{1,2}_).*(?=.pdf)', file_name)[0] for file_name in folder]
        elif journal == 'ASR':
            titles = [re.findall(r'(?<=ASR_\d{4}_\d{1,3}_\d{1,2}_).*(?=.pdf)', file_name)[0] for file_name in folder]
        title_words = [i for i in titles[indx].upper().split() if i not in stop_words and not i.isdigit()]

        ### Then add them to stop words & other hard-coded regularly occuring words
        non_section_words = hard_coded_non_section_words + title_words + stop_words
        matches = re.finditer(r'[A-Z]{4,}(\s+?[A-Z0-9]{2,}){0,}', string)

        if matches is not None:
            sections = dict()
            for match in matches:
                if match.group() not in non_section_words and match not in sections:
                    sections[match.group()] = match.span()
        for section in sections:
            outString += 'Section header "{}" found at {}'.format(section, sections[section]) + '\n'

        # Add in the article itself
        outString += '\n-----------\n||Article||\n-----------\n'
        if article is not None:
            outString += article
        elif article is None: #If regex wasn't able to split it according to the JSTOR access message just put the full article
            outString += string
        # Append the newly made string to the list. Again, there should be 1 per pdf
        out.append(outString)
    return out
    
def writeOut(out, pdfFolder = None, outFolder = None):
    for indx, file in tqdm(enumerate(out)):
        writeFilePath = 'corpus/{}/{}.txt'.format(outFolder, pdfFolder[indx][:-4])
        with open(writeFilePath, 'w') as f:
            f.write(file)
    print('done!')

def highlight(pattern, text, print_output=True):
    output = text
    lookforward = 0
    for match in pattern.finditer(text):
        start, end = match.start() + lookforward, match.end() + lookforward
        output = output[:start] + Back.YELLOW + Style.BRIGHT + output[start:end] + Style.RESET_ALL + output[end:]
        lookforward = len(output) - len(text)  

    if print_output:
        print(output)
    else:
        return output


# ## American Journal of Sociology articles

# In[ ]:


get_ipython().run_cell_magic('time', '', "# AJS articles - split into 3 periods\n\npre1946 = '/Users/Praveens/Desktop/ishan/Language-of-Science/articles/AJS pdf files/pre1946/'\npre1946pdfs = os.listdir(pre1946) # list of all the pdf files \npre1946pdfs.sort() # sort by year (and title)\n\nl946to1966 = '/Users/Praveens/Desktop/ishan/Language-of-Science/articles/AJS pdf files/1946to1966/'\nl946to1966pdfs = os.listdir(l946to1966) # list of all the pdf files\nl946to1966pdfs.sort()\n\npost1971 = '/Users/Praveens/Desktop/ishan/Language-of-Science/articles/AJS pdf files/post1971/'\npost1971pdfs = os.listdir(post1971) # list of all the pdf files\npost1971pdfs.sort()\n\n# Convert the articles to strings... this is the time-consuming step\nconvertedStrings_pre1946 = [PDFtoString(os.path.join(pre1946, file)) for file in tqdm(pre1946pdfs)]\nconvertedStrings_1946to1966 = [PDFtoString(os.path.join(l946to1966, file)) for file in tqdm(l946to1966pdfs)]\nconvertedStrings_post1971 = [PDFtoString(os.path.join(post1971, file)) for file in tqdm(post1971pdfs)]")


# ### String pre-processing

# In[ ]:


# Remove the double lines / extra space characters & footer download / use notice
convertedStrings_pre1946 = [' '.join(re.split('\n\n+', string)) for string in convertedStrings_pre1946]
pattern = re.compile(r'This content downloaded from (.)+\n(.)+\x0c')
#highlight(pattern, convertedStrings_pre1946[4])
convertedStrings_pre1946 = [' '.join(re.split(r'This content downloaded from (.)+\n(.)+\x0c', string)) for string in convertedStrings_pre1946]
printMetaInfo(convertedStrings_pre1946, pdfPaths = pre1946pdfs)


# In[ ]:


convertedStrings_1946to1966 = [' '.join(re.split('\n\n+', string)) for string in convertedStrings_1946to1966]
pattern = re.compile(r'[A-Z]{4,}(\s+?[A-Z0-9]{2,}){0,}')
highlight(pattern, convertedStrings_1946to1966[10])
convertedStrings_1946to1966 = [' '.join(re.split(r'This content downloaded from (.)+\n(.)+\x0c', string)) for string in convertedStrings_1946to1966]
printMetaInfo(convertedStrings_1946to1966, pdfPaths = l946to1966pdfs)


# In[ ]:


convertedStrings_post1971 = [' '.join(re.split('\n\n+', string)) for string in convertedStrings_post1971]
convertedStrings_post1971 = [' '.join(re.split(r'This content downloaded from (.)+\n(.)+\x0c', string)) for string in convertedStrings_post1971]
pattern = re.compile(r'[A-Z]{4,}(\s+?[A-Z0-9]{2,}){0,}')
highlight(pattern, convertedStrings_post1971[2])
printMetaInfo(convertedStrings_post1971, pdfPaths = post1971pdfs)


# ### Writing the files out to .txt files

# In[ ]:


out_pre1946 = createOutputStrings(convertedStrings_pre1946, folder = pre1946pdfs)
out_1946to1966 = createOutputStrings(convertedStrings_1946to1966, folder = l946to1966pdfs)
out_post1971 = createOutputStrings(convertedStrings_post1971, folder = post1971pdfs)

writeOut(out_pre1946, pdfFolder = pre1946pdfs, outFolder = 'AJS_pre1946')
writeOut(out_1946to1966, pdfFolder = l946to1966pdfs, outFolder = 'AJS_1946to1966')
writeOut(out_post1971, pdfFolder = post1971pdfs, outFolder = 'AJS_post1971')


# ## American Sociological Review articles

# In[ ]:


ASRpre1946 = '/Users/Praveens/Desktop/ishan/Language-of-Science/articles/ASR pdf files/pre1946/'
ASRpre1946pdfs = os.listdir(ASRpre1946) # list of all the pdf files 
ASRpre1946pdfs.sort() # sort by year (and title)

ASRpost1946 = '/Users/Praveens/Desktop/ishan/Language-of-Science/articles/ASR pdf files/post1946/'
ASRpost1946pdfs = os.listdir(ASRpost1946) # list of all the pdf files
ASRpost1946pdfs.sort()

# Convert the articles to strings... this is the time-consuming step
convertedStrings_ASRpre1946 = [PDFtoString(os.path.join(ASRpre1946, file)) for file in tqdm(ASRpre1946pdfs)]
convertedStrings_ASRpost1946 = [PDFtoString(os.path.join(ASRpost1946, file)) for file in tqdm(ASRpost1946pdfs)]


# In[ ]:


# Remove the double lines / extra space characters & footer download / use notice
convertedStrings_ASRpre1946 = [' '.join(re.split('\n\n+', string)) for string in convertedStrings_ASRpre1946]
convertedStrings_ASRpre1946 = [' '.join(re.split(r'This content downloaded from (.)+\n(.)+\x0c', string)) for string in convertedStrings_ASRpre1946]
printMetaInfo(convertedStrings_ASRpre1946, pdfPaths = ASRpre1946pdfs, journal = 'ASR')


# In[ ]:


convertedStrings_ASRpost1946 = [' '.join(re.split('\n\n+', string)) for string in convertedStrings_ASRpost1946]
convertedStrings_ASRpost1946 = [' '.join(re.split(r'This content downloaded from (.)+\n(.)+\x0c', string)) for string in convertedStrings_ASRpost1946]
printMetaInfo(convertedStrings_ASRpost1946, pdfPaths = ASRpost1946pdfs, journal = 'ASR')


# In[ ]:


out_ASRpre1946 = createOutputStrings(convertedStrings_ASRpre1946, folder = ASRpre1946pdfs, journal = 'ASR')
out_ASRpost1946 = createOutputStrings(convertedStrings_ASRpost1946, folder = ASRpost1946pdfs, journal = 'ASR')

writeOut(out_ASRpre1946, pdfFolder = ASRpre1946pdfs, outFolder = 'ASR_pre1946')
writeOut(out_ASRpost1946, pdfFolder = ASRpost1946pdfs, outFolder = 'ASR_post1946')

