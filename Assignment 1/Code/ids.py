###Project 1 COMP 6970 Adversarial Machine Learning Fall 2020
###Team Members - Kushagra Kushagra, Mousumi Akter, Chase Townson

from bs4 import BeautifulSoup, SoupStrainer
import httplib2
from collections import defaultdict
import numpy as np
import sys
sys.setrecursionlimit(100000)

global maxDepth
global userURL
nodeNumber = 0
global node
outputstr = ''
printedNode = []
urlStack = []

f = open("demofile.txt", "a")

def push(nodeNumber, urlStack):
    urlStack.insert(0, nodeNumber)

def get(urlStack):
    return urlStack[0]


def graphGenerate(node):
    global nodeNumber
    depth = node[4]
    urlgraph.append(node)
    if depth == maxDepth:
        return
    childIndex = -1
    for link in BeautifulSoup(response, parse_only=SoupStrainer('a'), features="html.parser"):
        if 'http' in link['href']:
            nodeNumber = nodeNumber+1
            childIndex = childIndex+1
            graphGenerate([nodeNumber, link['href'], childIndex, node[0], node[4]+1])

#node = [nodeNumber, nodeURL, childNodeIndex, parentNodeIndex, depth]

def graphPrintids():
    global outputstr

    if len(urlStack) == 0:
        print("[]")
        return
    nonum = get(urlStack)
    for elem in range(0, len(urlgraph)):
        if urlgraph[elem][0] == nonum:
            print(urlgraph[elem])
            #outputstr += str(urlgraph[elem]) + '\n'

            ### HTML extract as File Here with Feature Vector



            ######################


            urlStack.pop(0)
    for elem in range(len(urlgraph)-1, 0, -1):
        if urlgraph[elem][3] == nonum:
            push(urlgraph[elem][0], urlStack)

    graphPrintids()



if __name__ == "__main__":
    maxDepth = 3
    userURL = "http://www.google.com"
    http = httplib2.Http()
    status, response = http.request(userURL)

    
    urlgraph = []
    visitedlinks = set()
    nodeNumber = 0
    nodeURL = userURL
    childNodeIndex = 0
    parentNodeIndex = -1
    depth = 0
    node = [nodeNumber, nodeURL, childNodeIndex, parentNodeIndex, depth]
    graphGenerate(node)
    nodeNumber = 0
    push(nodeNumber, urlStack)
    #graphPrint(urlgraph)
    #print(urlgraph)
    graphPrintids()
    #f.write(outputstr)
    #f.close()