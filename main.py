from NaiveBayes import NaiveBayes

def main():
    nb = NaiveBayes()
    text = 'A bad place, really terrible'
    nb.evaluate()
    #print(nb.predictSentence(text))


if __name__ == '__main__':
    main()