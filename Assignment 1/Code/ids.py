###Project 1 COMP 6970 Adversarial Machine Learning Fall 2020
###Team Members - Kushagra Kushagra, Mousumi Akter, Chase Townson

from bs4 import BeautifulSoup, SoupStrainer
import httplib2
from collections import defaultdict
import sys
sys.setrecursionlimit(100000)

global maxDepth
global userURL
global node
nodeNumber = 0
outputstr = ''
printedNode = []
urlStack = []

def push(nodeNumber, urlStack):
    urlStack.insert(0, nodeNumber)

def get(urlStack):
    return urlStack[0]

#node = [nodeNumber, nodeURL, childNodeIndex, parentNodeIndex, depth]

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


def graphPrintids():
    global outputstr
    global printedNode

    if len(urlStack) == 0:
        print("[]")
        return
    nonum = get(urlStack) #first element from urlStack
    repeatedNodeFound = 0
    for elem in range(0, len(urlgraph)):
        if urlgraph[elem][0] == nonum:

            if urlgraph[elem][1] not in printedNode:
                # print(urlgraph[elem])
                printedNode.append(urlgraph[elem][1])
                ### HTML extract as File Here with Feature Vector
                ######################
            else:
                repeatedNodeFound = 1

            urlStack.pop(0)

    if repeatedNodeFound == 0:
        for elem in range(len(urlgraph)-1, 0, -1):
            if urlgraph[elem][3] == nonum:
                if urlgraph[elem][1] not in printedNode:
                    push(urlgraph[elem][0], urlStack)

    graphPrintids()



if __name__ == "__main__":

    f = open("demofile.txt", "w")

    maxDepth = int(input("Enter max depth: "))
    userURL = input("Enter URL: ")

    http = httplib2.Http()
    status, response = http.request(userURL)

    
    urlgraph = []
    nodeNumber = 0
    nodeURL = userURL
    childNodeIndex = 0
    parentNodeIndex = -1
    depth = 0
    printedNode = []
    node = [nodeNumber, nodeURL, childNodeIndex, parentNodeIndex, depth]
    graphGenerate(node)
    nodeNumber = 0
    push(nodeNumber, urlStack)
    graphPrintids()

    for i in range(len(printedNode)):
        f.write(printedNode[i]+'\n')    

