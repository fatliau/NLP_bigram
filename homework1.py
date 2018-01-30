#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 14:55:02 2018

@author: JC (Chieh-Chi Chen, cxc166530 UTD)

http://www.hlt.utdallas.edu/~moldovan/CS6320.18S/Homework1.pdf

CS 6320 Natural Language Processing Spring 2018
Homework 1.
Due Jan. 31, 2018 at 2:30 pm
An automatic speech recognition system has provided two written sentences as possible interpretations to a speech input.
S1: Apple computer is the first product of the company .
S2: Apple introduced the new version of iphone in 2008 .
Using the bigram language model trained on Corpus, find out which of the two sentences is more probable. Compute the probability of each of the two sentences under the two following scenarios:
i. Use the bigram model without smoothing.
ii. Use the bigram model with add-one smoothing
Write a computer program to:
A. For each of the two scenarios, construct the tables with the bigram counts for the two sentences above.
B. For each of the two scenarios, construct the table with the bigram probabilities for the sentences.
C. For each of the two scenarios, compute the total probabilities for each sentence S1 and S2.
What to turn in:
Your code and a Readme file for compiling the code. The Readme file should contain a command line that can be used to compile and execute your program directly. For example: python homework1.py <corpus.txt> <sentence1> <sentence2>
The output of the program should contain:
8 tables: the bigram counts table and bigram probability table of the two sentences under two scenarios.
4 probabilities: the total probabilities of the two sentences under two scenarios.
"""

from collections import OrderedDict
from nltk import tokenize 
import pandas as pd  
import sys


def trainingWords(file):
    f = open(file, 'r')
    message = f.read()
    words = tokenize.word_tokenize(message)
    return words

def countingTable(sentence, trainingSet):
    tokenWords = tokenize.word_tokenize(sentence)
    table = OrderedDict({tokenWord : 0 for tokenWord in tokenWords})
    for word in trainingSet:
        if word in table.keys():
            table[word] += 1
    return table

def countingBigramTable(sentence, trainingSet):
    tokenWords = tokenize.word_tokenize(sentence)
    doubleTable = OrderedDict({tokenWord : OrderedDict({}) for tokenWord in tokenWords})
    for verticalItem in doubleTable.keys():
        doubleTable[verticalItem] = {tokenWord : 0 for tokenWord in tokenWords}
    for i, ele in enumerate(trainingSet):
        if ele in doubleTable.keys():
            if i < len(trainingSet)-1:
                if trainingSet[i+1] in doubleTable[ele].keys():
                    doubleTable[ele][trainingSet[i+1]] += 1
    return doubleTable
    
def countingBigramTableSmooth(sentence, trainingSet):
    tokenWords = tokenize.word_tokenize(sentence)
    doubleTable = OrderedDict({tokenWord : OrderedDict({}) for tokenWord in tokenWords})
    for verticalItem in doubleTable.keys():
        #Smotth: initialiting all value from 1
        doubleTable[verticalItem] = {tokenWord : 1 for tokenWord in tokenWords}
    for i, ele in enumerate(trainingSet):
        if ele in doubleTable.keys():
            if i < len(trainingSet)-1:
                if trainingSet[i+1] in doubleTable[ele].keys():
                    doubleTable[ele][trainingSet[i+1]] += 1
    return doubleTable      

def bigramProb(sentence, trainingSet):
    uniTable = countingTable(sentence, trainingSet)
    biTable = countingBigramTable(sentence, trainingSet)
    for verKey in biTable.keys():
        for horKey in biTable[verKey].keys():
            if horKey in uniTable.keys():
                if uniTable[horKey] == 0:
                    biTable[verKey][horKey] = - float('inf')
                else:
                    biTable[verKey][horKey] = biTable[verKey][horKey]/uniTable[horKey]
            else:
                print("WARNING: column mismatch")
    return biTable
    
def bigramProbSmooth(sentence, trainingSet):
    uniTable = countingTable(sentence, trainingSet)
    biTable = countingBigramTable(sentence, trainingSet)
    V = len(set(trainingSet))
    for verKey in biTable.keys():
        for horKey in biTable[verKey].keys():
            if horKey in uniTable.keys():
                biTable[verKey][horKey] = (biTable[verKey][horKey]+1) / (uniTable[horKey]+V)
            else:
                print("WARNING: column mismatch")
    return biTable

def wholeSentence(sentence, trainingSet):
    tokenWords = tokenize.word_tokenize(sentence)
    biTable = bigramProb(sentence, trainingSet)
    prob = 1
    count = 0
    for i, word in enumerate(tokenWords):
        if i < len(tokenWords)-1:
            if biTable[word][tokenWords[i+1]] != - float('inf'):
                prob = prob * (biTable[word][tokenWords[i+1]])
                count += 1
    print("did multiplied: ", count, " compare length: ", len(tokenWords)-1)
    print("ignored ",len(tokenWords)-1-count," probabilities due to zero frequency")
    return prob

def wholeSentenceSmooth(sentence, trainingSet):
    tokenWords = tokenize.word_tokenize(sentence)
    biTable = bigramProbSmooth(sentence, trainingSet)
    prob = 1
    count = 0
    for i, word in enumerate(tokenWords):
        if i < len(tokenWords)-1:
            prob = prob * (biTable[word][tokenWords[i+1]])
            count += 1
    print("did multiplied: ", count, " compare length: ", len(tokenWords)-1)
    return prob
        

def main():
    #make pd.DataFrame not be divided to next line
    pd.set_option('expand_frame_repr', False)
    if len(sys.argv) != 4:
        print("Argument format dismatch! Correct form as: python homework1.py <corpus.txt> <sentence1> <sentence2>")
    else:
        words = trainingWords(sys.argv[1])
        S1 = sys.argv[2]
        S2 = sys.argv[3]
        print("\nbigram counts table for Sentence 1")
        BigramTableS1 = countingBigramTable(S1,words)
        print(pd.DataFrame(BigramTableS1, index=BigramTableS1[next(iter(BigramTableS1.keys()))]))
        print("\nbigram counts table for Sentence 2")
        BigramTableS2 = countingBigramTable(S2,words)
        print(pd.DataFrame(BigramTableS2, index=BigramTableS2[next(iter(BigramTableS2.keys()))]))
        print("\nbigram probability table for Sentence 1")
        BigramProbS1 = bigramProb(S1, words)
        print(pd.DataFrame(BigramProbS1, index=BigramProbS1[next(iter(BigramProbS1.keys()))]))
        print("\nbigram probability table for Sentence 2")
        BigramProbS2 = bigramProb(S2, words)
        print(pd.DataFrame(BigramProbS2, index=BigramProbS2[next(iter(BigramProbS2.keys()))]))
        print("\nbigram counts table for Sentence 1 with smoothing")
        BigramTableSmoothS1 = countingBigramTableSmooth(S1,words)
        print(pd.DataFrame(BigramTableSmoothS1, index=BigramTableSmoothS1[next(iter(BigramTableSmoothS1.keys()))]))
        print("\nbigram counts table for Sentence 2 with smoothing")
        BigramTableSmoothS2 = countingBigramTableSmooth(S2,words)
        print(pd.DataFrame(BigramTableSmoothS2, index=BigramTableSmoothS2[next(iter(BigramTableSmoothS2.keys()))]))
        print("\nbigram probability table for Sentence 1 with smoothing")
        BigramProbSmoothS1 = bigramProbSmooth(S1, words)
        print(pd.DataFrame(BigramProbSmoothS1, index=BigramProbSmoothS1[next(iter(BigramProbSmoothS1.keys()))]))
        print("\nbigram probability table for Sentence 2 with smoothing")
        BigramProbSmoothS2 = bigramProbSmooth(S2, words)
        print(pd.DataFrame(BigramProbSmoothS2, index=BigramProbSmoothS2[next(iter(BigramProbSmoothS2.keys()))]))
        print("\nthe total probabilities for Sentence 1")
        print(wholeSentence(S1, words))
        print("\nthe total probabilities for Sentence 2")
        print(wholeSentence(S2, words))
        print("\nthe total probabilities for Sentence 1 with smoothing")
        probS1 = wholeSentenceSmooth(S1, words)
        print(probS1)
        print("\nthe total probabilities for Sentence 2 with smoothing")
        probS2 = wholeSentenceSmooth(S2, words)
        print(probS2)
        if probS1 <= probS2:
            print("\nSentence 2 is better, which is \"",S2,"\"")
        else:
            print("\nSentence 1 is better, which is \"",S1,"\"")


if __name__ == '__main__':
    main()


#python homework1.py 'corpus.txt' "Apple computer is the first product of the company" "Apple introduced the new version of iPhone in 2008"


