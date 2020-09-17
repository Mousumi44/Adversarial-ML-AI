from __future__ import division, unicode_literals 
from bs4 import BeautifulSoup
import requests
import codecs
from nltk.tokenize import word_tokenize
from nltk import tokenize
import re
import os.path
import shutil


def link_to_html(url, fname):

    source  = requests.get(url).text

    soup = BeautifulSoup(source, 'lxml')

    html_str = soup.prettify()

    # print(html_str)

    fname = os.path.join('./HTML_FILE/', fname)

    Html_file= open(fname,"w")
    Html_file.write(html_str)
    Html_file.close()

def clean_html(fname):
    fname = os.path.join('./HTML_FILE/', fname)  
    f=codecs.open(fname, 'r', 'utf-8')
    document = BeautifulSoup(f.read(),features = "lxml").get_text()
    # return document #if don't want preprocessing

    # Preprocessing
    docwords=word_tokenize(document)

    st = ""
    for line in docwords:
        line = (line.rstrip())
        if line and re.match("^[A-Za-z]*$",line) and len(line)>1:
            st=st+""+line
    return st

def char_unigram_feature(st):
    """
    st: clean html , type: str
    return dictionary of character count
    """
    dic = {}
    for l in st:
        try:
            dic[l] +=1
        except KeyError:
            dic[l] = 1

    return dic

if __name__ == '__main__':

    ref_path = './HTML_FILE/'
    if not os.path.exists(ref_path):
        os.mkdir(ref_path)

    

    with open ('demofile.txt') as f:
        sample = 0
        for line in f:
            sample+=1
            url = line
            fname = 'file'+str(sample)+'.html'
            try:
                link_to_html(url, fname)
                st = clean_html(fname)
                print(char_unigram_feature(st))
                print('\n')
            except:
                print("Exception Occured\n")

    shutil.rmtree(ref_path)