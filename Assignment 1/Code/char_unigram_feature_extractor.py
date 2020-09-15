from __future__ import division, unicode_literals 
from bs4 import BeautifulSoup
import requests
import codecs
import nltk
from nltk.tokenize import word_tokenize
from nltk import tokenize
import re
import string


def link_to_html(url, fname):

    source  = requests.get(url).text

    soup = BeautifulSoup(source, 'lxml')

    html_str = soup.prettify()

    # print(html_str)

    Html_file= open(fname,"w")
    Html_file.write(html_str)
    Html_file.close()

def clean_html(fname):  
    f=codecs.open(fname, 'r', 'utf-8')
    document = BeautifulSoup(f.read(),features = "lxml").get_text()

    docwords=word_tokenize(document)

    st = ""
    for line in docwords:
        line = (line.rstrip())
        if line and re.match("^[A-Za-z]*$",line) and len(line)>1:
            st=st+" "+line
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


url = 'http://auburn.edu/'
fname = 'au.html'

link_to_html(url, fname)
st = clean_html(fname)
print(char_unigram_feature(st))