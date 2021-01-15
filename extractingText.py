#!/usr/bin/env python
# coding: utf-8

# ### Imports, Reading in files & Defining functions

# In[51]:


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

from autocorrect import Speller # Spell checker
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
        matches = re.finditer(r'FIG|FIGURE|Fig', string)
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
        figure_matches = re.finditer(r'FIG|FIGURE|Fig', string)
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
        matches = re.finditer(r'[A-Z]{4,}(\s+?[A-Z]{2,}){0,}', string)

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

def highlight(pattern, text, printOut = True):
    output = text
    lookforward = 0
    for match in pattern.finditer(text):
        start, end = match.start() + lookforward, match.end() + lookforward
        output = output[:start] + Back.YELLOW + Style.BRIGHT + output[start:end] + Style.RESET_ALL + output[end:]
        lookforward = len(output) - len(text)  

    if printOut:
        print(output)
    else:
        return output


# In[2]:


with open('example.txt', 'r') as f:
    d = f.read()
pattern = re.compile(r'[A-Z]{4,}(\s+?[A-Z]{2,}){0,}')
highlight(pattern, d)


# ## American Journal of Sociology articles

# In[3]:


get_ipython().run_cell_magic('time', '', "# AJS articles - split into 3 periods\n\npre1946 = '/Users/Praveens/Desktop/ishan/Language-of-Science/articles/AJS pdf files/pre1946/'\npre1946pdfs = os.listdir(pre1946) # list of all the pdf files \npre1946pdfs.sort() # sort by year (and title)\n\nl946to1966 = '/Users/Praveens/Desktop/ishan/Language-of-Science/articles/AJS pdf files/1946to1966/'\nl946to1966pdfs = os.listdir(l946to1966) # list of all the pdf files\nl946to1966pdfs.sort()\n\npost1971 = '/Users/Praveens/Desktop/ishan/Language-of-Science/articles/AJS pdf files/post1971/'\npost1971pdfs = os.listdir(post1971) # list of all the pdf files\npost1971pdfs.sort()\n\n# Convert the articles to strings... this is the time-consuming step\nconvertedStrings_pre1946 = [PDFtoString(os.path.join(pre1946, file)) for file in tqdm(pre1946pdfs) if file[-4:] == '.pdf']\nconvertedStrings_1946to1966 = [PDFtoString(os.path.join(l946to1966, file)) for file in tqdm(l946to1966pdfs) if file[-4:] == '.pdf']\nconvertedStrings_post1971 = [PDFtoString(os.path.join(post1971, file)) for file in tqdm(post1971pdfs) if file[-4:] == '.pdf']")


# ### String pre-processing

# In[4]:


# Remove the double lines / extra space characters & footer download / use notice
convertedStrings_pre1946 = [' '.join(re.split('\n\n+', string)) for string in convertedStrings_pre1946]
pattern = re.compile(r'This content downloaded from (.)+\n(.)+\x0c')
#highlight(pattern, convertedStrings_pre1946[4])
convertedStrings_pre1946 = [' '.join(re.split(r'This content downloaded from (.)+\n(.)+\x0c', string)) for string in convertedStrings_pre1946]
#printMetaInfo(convertedStrings_pre1946, pdfPaths = pre1946pdfs)


# In[5]:


convertedStrings_1946to1966 = [' '.join(re.split('\n\n+', string)) for string in convertedStrings_1946to1966]
pattern = re.compile(r'[A-Z]{4,}(\s+?[A-Z0-9]{2,}){0,}')
#highlight(pattern, convertedStrings_1946to1966[10])
convertedStrings_1946to1966 = [' '.join(re.split(r'This content downloaded from (.)+\n(.)+\x0c', string)) for string in convertedStrings_1946to1966]
#printMetaInfo(convertedStrings_1946to1966, pdfPaths = l946to1966pdfs)


# In[6]:


convertedStrings_post1971 = [' '.join(re.split('\n\n+', string)) for string in convertedStrings_post1971]
convertedStrings_post1971 = [' '.join(re.split(r'This content downloaded from (.)+\n(.)+\x0c', string)) for string in convertedStrings_post1971]
pattern = re.compile(r'[A-Z]{4,}(\s+?[A-Z0-9]{2,}){0,}')
#highlight(pattern, convertedStrings_post1971[2])
#printMetaInfo(convertedStrings_post1971, pdfPaths = post1971pdfs)


# ### Writing the files out to .txt files

# In[7]:


out_pre1946 = createOutputStrings(convertedStrings_pre1946, folder = pre1946pdfs)
out_1946to1966 = createOutputStrings(convertedStrings_1946to1966, folder = l946to1966pdfs)
out_post1971 = createOutputStrings(convertedStrings_post1971, folder = post1971pdfs)

writeOut(out_pre1946, pdfFolder = pre1946pdfs, outFolder = 'AJS_pre1946')
writeOut(out_1946to1966, pdfFolder = l946to1966pdfs, outFolder = 'AJS_1946to1966')
writeOut(out_post1971, pdfFolder = post1971pdfs, outFolder = 'AJS_post1971')


# ## American Sociological Review articles

# In[8]:


ASRpre1946 = '/Users/Praveens/Desktop/ishan/Language-of-Science/articles/ASR pdf files/pre1946/'
ASRpre1946pdfs = os.listdir(ASRpre1946) # list of all the pdf files 
ASRpre1946pdfs.sort() # sort by year (and title)

ASRpost1946 = '/Users/Praveens/Desktop/ishan/Language-of-Science/articles/ASR pdf files/post1946/'
ASRpost1946pdfs = os.listdir(ASRpost1946) # list of all the pdf files
ASRpost1946pdfs.sort()

# Convert the articles to strings... this is the time-consuming step
convertedStrings_ASRpre1946 = [PDFtoString(os.path.join(ASRpre1946, file)) for file in tqdm(ASRpre1946pdfs)]
convertedStrings_ASRpost1946 = [PDFtoString(os.path.join(ASRpost1946, file)) for file in tqdm(ASRpost1946pdfs)]


# In[9]:


# Remove the double lines / extra space characters & footer download / use notice
convertedStrings_ASRpre1946 = [' '.join(re.split('\n\n+', string)) for string in convertedStrings_ASRpre1946]
convertedStrings_ASRpre1946 = [' '.join(re.split(r'This content downloaded from (.)+\n(.)+\x0c', string)) for string in convertedStrings_ASRpre1946]
#printMetaInfo(convertedStrings_ASRpre1946, pdfPaths = ASRpre1946pdfs, journal = 'ASR')


# In[10]:


convertedStrings_ASRpost1946 = [' '.join(re.split('\n\n+', string)) for string in convertedStrings_ASRpost1946]
convertedStrings_ASRpost1946 = [' '.join(re.split(r'This content downloaded from (.)+\n(.)+\x0c', string)) for string in convertedStrings_ASRpost1946]
#printMetaInfo(convertedStrings_ASRpost1946, pdfPaths = ASRpost1946pdfs, journal = 'ASR')


# In[11]:


out_ASRpre1946 = createOutputStrings(convertedStrings_ASRpre1946, folder = ASRpre1946pdfs, journal = 'ASR')
out_ASRpost1946 = createOutputStrings(convertedStrings_ASRpost1946, folder = ASRpost1946pdfs, journal = 'ASR')

writeOut(out_ASRpre1946, pdfFolder = ASRpre1946pdfs, outFolder = 'ASR_pre1946')
writeOut(out_ASRpost1946, pdfFolder = ASRpost1946pdfs, outFolder = 'ASR_post1946')


# ### Consolidating PyTesseract output

# In[12]:


# First, for a few test cases - the first two files of AJS_pre1946
import shutil

tess_bF = 'tesseract-corpus-raw'
journalFolder= 'AJS_pre1946'

# For each folder in the journalFolders, find the out.txt file and copy it to the corpus w/ the name of the folder
for folder in os.listdir(os.path.join(tess_bF, journalFolder))[:2]: # First two files for now
    for file in os.listdir(os.path.join(tess_bF, journalFolder, folder)):
        if file[-4:] == '.txt':
            outFile = open(os.path.join('tesseract-corpus', journalFolder, folder) + '.txt', 'w')
            srcFile = open(os.path.join(tess_bF, journalFolder, folder, file), 'r')
            shutil.copyfileobj(srcFile, outFile)


# In[13]:


# Then, cycle through each journal and do the same
journalFolders = ['AJS_pre1946', 'ASR_pre1946', 'ASR_post1946', 'AJS_1946to1966', 'AJS_post1971']

# For each folder in the journalFolders, find the out.txt file and copy it to the corpus w/ the name of the folder
for journal in journalFolders:
    for folder in os.listdir(os.path.join(tess_bF, journal)):
        if folder[0] != '.':
            for file in os.listdir(os.path.join(tess_bF, journal, folder)):
                if file[-4:] == '.txt':
                    outFile = open(os.path.join('tesseract-corpus', journal, folder) + '.txt', 'w')
                    srcFile = open(os.path.join(tess_bF, journal, folder, file), 'r')
                    shutil.copyfileobj(srcFile, outFile) 
                    
# Resulting output is the corpus .txt files in tesseract-corpus nicely sectioned into each of the journal folders


# ### Adding headers back into the PyTesseract output

# In[19]:


os.listdir('tesseract-corpus')

for journal in journalFolders[:1]: # [AJS_pre1946, ASR_pre1946, ASR_post1946, AJS_1946to1966, AJS_post1971]
    convertedTxtFiles = os.listdir(os.path.join('tesseract-corpus', journal))
    convertedTxtFiles.sort()
    for indx, file in enumerate(convertedTxtFiles[:5]): # First five files
        print(indx, file)
        tess_file = open(os.path.join('tesseract-corpus', journal, file), 'r').read()


# In[32]:


# Example tesseract output text file
#print(convertedTxtFiles[0])
tess_file = open(os.path.join('tesseract-corpus', 'AJS_pre1946', convertedTxtFiles[0]), 'r').read()
print(tess_file)


# ### We want to repurpose the section headers - we don't care about the indices so much as we do about the words around it

# In[49]:


journal = 'AJS'
folder = pre1946pdfs
string = convertedStrings_pre1946[1]
outString = ''
### First, look for the title words and tokenize them.
if journal == 'AJS':
    titles = [re.findall(r'(?<=AJS_\d{4}_\d{1,3}_\d{1,2}_).*(?=.pdf)', file_name)[0] for file_name in folder]
elif journal == 'ASR':
    titles = [re.findall(r'(?<=ASR_\d{4}_\d{1,3}_\d{1,2}_).*(?=.pdf)', file_name)[0] for file_name in folder]
title_words = [i for i in titles[indx].upper().split() if i not in stop_words and not i.isdigit()]

### Then add them to stop words & other hard-coded regularly occuring words
non_section_words = hard_coded_non_section_words + title_words + stop_words
matches = re.finditer(r'[A-Z]{4,}(\s+?[A-Z]{2,}){0,}', string)

if matches is not None:
    sections = dict()
    for match in matches:
        if match.group() not in non_section_words and match not in sections:
            sections[match.group()] = match.span()
for section in sections:
    outString += 'Section header "{}" found at {}'.format(section, sections[section]) + '\n'


# In[53]:


get_ipython().run_cell_magic('time', '', "spell = Speller('en')\nspell(string)")


# In[50]:


matches = re.finditer(r'[A-Z]{4,}(\s+?[A-Z]{2,}){0,}', string)
if matches is not None:
    sections = dict()
    for match in matches:
        if match.group() not in non_section_words and match not in sections:
            sections[match.group()] = match.span()
            print(match.group(), match.span())


# In[34]:


# Add section info 

        ### First, look for the title words and tokenize them.
        if journal == 'AJS':
            titles = [re.findall(r'(?<=AJS_\d{4}_\d{1,3}_\d{1,2}_).*(?=.pdf)', file_name)[0] for file_name in folder]
        elif journal == 'ASR':
            titles = [re.findall(r'(?<=ASR_\d{4}_\d{1,3}_\d{1,2}_).*(?=.pdf)', file_name)[0] for file_name in folder]
        title_words = [i for i in titles[indx].upper().split() if i not in stop_words and not i.isdigit()]

        ### Then add them to stop words & other hard-coded regularly occuring words
        non_section_words = hard_coded_non_section_words + title_words + stop_words
        matches = re.finditer(r'[A-Z]{4,}(\s+?[A-Z]{2,}){0,}', string)

        if matches is not None:
            sections = dict()
            for match in matches:
                if match.group() not in non_section_words and match not in sections:
                    sections[match.group()] = match.span()
        for section in sections:
            outString += 'Section header "{}" found at {}'.format(section, sections[section]) + '\n'


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[20]:


string = out_pre1946[0] # Example string

# Split string into header tag and main article
header_tag, article_header, main_article = re.split('(-----------\n\|\|Article\|\|\n-----------)', string)
article = article_header + main_article

section_pattern = re.compile(r'(?<=Section header ") ?([A-Z]*\s*){1,}.*(?=" found at)')
section_location = re.compile(r'(\d{1,}+, \d{1,}+)')
sections = dict()
for match_pair in zip(section_pattern.finditer(header_tag), section_location.finditer(header_tag)):
    sections[match_pair[0].group()] = [int(indx) for indx in match_pair[1].group().split(', ')]
for section, section_indices in sections.items():
    print(section, section_indices)


# In[223]:


for section, section_indices in sections.items():
    pattern = re.compile(section)
    smallest_indx_diff = 1e5
    for counter, match in enumerate(pattern.finditer(string)):
        print(counter, match)
        if np.abs(section_indices[0] - match.span()[0]) < smallest_indx_diff:
            smallest_indx_diff = np.abs(section_indices[0] - match.span()[0])
    for indx, match in enumerate(pattern.finditer(string)):
        if indx == counter:
            print(match.group())
            print(re.search(re.compile(match.group()), article))


# In[224]:


re.search(re.compile(match.group()), article)


# In[200]:


pattern = re.compile('ANTI')
for match in pattern.finditer(string):
    print(match.group(), match.span(), match.span()[0] - len(header_tag), match.span()[1] - len(header_tag),
          match.span())


# In[225]:


highlight(re.compile('ANTI'), string)


# In[144]:


section_location = re.compile(r'(\d{1,}+, \d{1,}+)')
for match_pair in zip(section_pattern.finditer(header_tag), section_location.finditer(header_tag)):
    print(match_pair[0].group(), match_pair[1].group())


# In[16]:


string = out_pre1946[-7] # Example string - 
header_tag = string[:re.search(r'-----------\n\|\|Article\|\|\n-----------', string).span()[0]]
print(header_tag)

pattern = re.compile(r'Section header "?([A-Z]*\s*){1,}.*')
matches = pattern.finditer(header_tag)
for match in matches:
    print(match.group())


# In[499]:


non_section_words = hard_coded_non_section_words + stop_words
lookback = 30

for string in out_pre1946:
    header_tag = string[:re.search(r'-----------\n\|\|Article\|\|\n-----------', string).span()[0]]
    article = string[re.search(r'-----------\n\|\|Article\|\|\n-----------', string).span()[0]:]
    pattern = re.compile(r'Section header "?([A-Z]*\s*){1,}.*')
    
    matches = re.finditer(r'[A-Z]{4,}(\s+?[A-Z]{2,}){0,}', header_tag)
    for match in matches:
        if match.group() not in non_section_words and match not in sections:
            sections[match.group()] = match.span()
    compareIndices = list()
    for section in sections:
        match_indx = sections[section][0]
        match_spans = dict()
        substring = article[match_indx - lookback:match_indx]
        for indx, word in enumerate(substring.split()):
            print(substring.split())
            matches = re.finditer(word, string)
            match_spans[word] = np.array([match.span() for match in matches])
        for word in match_spans:
            counter = match_spans[word].shape[0] - 1
            if counter != -1:
                compareIndices.append(match_spans[word][counter])

    start_indices, end_indices = list(), list()
    for i in range(1, len(compareIndices)):
        if compareIndices[i-1][1] + 1 == compareIndices[i][0]:
            start_indx = compareIndices[i-1][0]
            end_indx = compareIndices[i][1]

            start_indices.append(start_indx)
            end_indices.append(end_indx)
    for indx1 in start_indices:
        for indx2 in end_indices:
            if indx2 - indx1 > 10 and indx2 - indx1 < 30:
                unique_identifier = string[indx1:indx2]

    print('Tag match found: {}'.format(re.search(unique_identifier, tess_file)))


# In[511]:


lookback = 30
compareIndices = list()
for string in out_pre1946:
    # Split article into header tag containing meta-information and the main article 
    header_tag = string[:re.search(r'-----------\n\|\|Article\|\|\n-----------', string).span()[0]]
    article = string[re.search(r'-----------\n\|\|Article\|\|\n-----------', string).span()[0]:]
    
    # Look for sections within the article
    matches = re.finditer(r'[A-Z]{4,}(\s+?[A-Z]{2,}){0,}', string)
    sections = dict()
    for match in matches:
        if match.group() not in non_section_words and match not in sections:
            sections[match.group()] = match.span()
    for section in sections:
        match_indx = sections[section][0]
        match_spans = dict()
        substring = string[match_indx - lookback:match_indx]
        print(substring.split())
        #for indx, word in enumerate(substring.split()):
        #    matches = re.finditer(word, string)
        #    match_spans[word] = np.array([match.span() for match in matches])
        #for word in match_spans:
        #    counter = match_spans[word].shape[0] - 1
        #    if counter != -1:
        #        compareIndices.append(match_spans[word][counter])


# In[477]:


lookback = 30
compareIndices = list()
for section in sections:
    match_indx = sections[section][0]
    match_spans = dict()
    substring = string[match_indx - lookback:match_indx]
    for indx, word in enumerate(substring.split()):
        matches = re.finditer(word, string)
        match_spans[word] = np.array([match.span() for match in matches])
    for word in match_spans:
        counter = match_spans[word].shape[0] - 1
        if counter != -1:
            compareIndices.append(match_spans[word][counter])

start_indices, end_indices = list(), list()
for i in range(1, len(compareIndices)):
    if compareIndices[i-1][1] + 1 == compareIndices[i][0]:
        start_indx = compareIndices[i-1][0]
        end_indx = compareIndices[i][1]

        start_indices.append(start_indx)
        end_indices.append(end_indx)
for indx1 in start_indices:
    for indx2 in end_indices:
        if indx2 - indx1 > 10 and indx2 - indx1 < 30:
            unique_identifier = string[indx1:indx2]

print('Tag match found: {}'.format(re.search(unique_identifier, tess_file)))

