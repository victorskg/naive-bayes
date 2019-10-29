import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class NaiveBayes(object):
    def __init__(self):
        self.dataSet = pd.read_csv("datasets/chennai_reviews.csv", header=None, sep=',', dtype=str).to_numpy()
        self.dataSet = self.prepareData(self.dataSet)
        self.train, self.test = train_test_split(self.dataSet, test_size=0.33)
        self.splited = self.splitByClass(self.train)
        self.wordsObject = self.splitWords(self.splited)
        self.countObject, self.totalUniqueWords = self.countOcurrences(self.wordsObject)

    def evaluate(self):
        hits = 0
        for data in self.test:
            predict, prob = self.predictSentence(data[2])
            hits = hits + 1 if (int(predict) == int(data[3])) else hits
        print('Acertou {0} de {1}'.format(hits, len(self.test)))


    def predictSentence(self, sentence):
        probabilitys = [self.probabilityOfSentence(sentence, i) for i in ['1', '2', '3']]
        maxValue = np.max(probabilitys)
        return probabilitys.index(maxValue) + 1, maxValue

    def probabilityOfSentence(self, sentence, classs):
        total = 0
        for word in sentence.split():
            total = total + self.probabilityOfWord(word, classs)
        return total

    def probabilityOfWord(self, word, classs):
        total = 0
        for key, value in self.countObject[classs].items():
            total = total + float(value) if (word in key) else total
        return float(total) / float(self.wordsObject[classs]['count'])

    @staticmethod
    def splitWords(dataSet):
        wordsObject = {}
        for classs, items in dataSet.items():
            if (classs not in wordsObject):
                wordsObject[classs] = {
                    'count': 0,
                    'words': []
                }
            for item in items:
                words = item[2].lower().split()
                wordsObject[classs]['words'] = wordsObject[classs]['words'] + words
                wordsObject[classs]['count'] = wordsObject[classs]['count'] + len(words)       
        return wordsObject

    @staticmethod
    def countOcurrences(wordsObj):
        obj = {}
        allWords = []
        for key, value in wordsObj.items():
            obj[key] = {}
            allWords = allWords + value['words']
            occurrences = np.array(np.unique(value['words'], return_counts=True)).T
            for item in occurrences:
                obj[key][item[0]] = item[1]
        totalUniqueWords = len(np.array(np.unique(allWords)))
        return obj, totalUniqueWords

    @staticmethod
    def prepareData(dataSet):
        validClasses = ['1', '2', '3']
        for i in range(len(dataSet)):
            data = dataSet[i]
            if (data[3] not in validClasses):
                index = 3
                text = data[2]
                while (data[index] not in validClasses):
                    txt = str(data[index])
                    text = text + txt if txt.startswith(' ') else text + ' ' + txt
                    index = index + 1
                data[2] = text
                data[3] = data[index]
        return dataSet

    @staticmethod
    def splitByClass(dataSet):
        splited = {}
        for i in range(len(dataSet)):
            data = dataSet[i]
            classs = data[3]
            if (classs not in splited):
                splited[classs] = []
            splited[classs].append(data)
        return splited
