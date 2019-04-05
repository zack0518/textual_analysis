from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from geotext import GeoText
from numpy import *
#Term weighting

keyWords = ["shark"]

def termWeighting(text):
    textChunk = word_tokenize(text)
    print(textChunk)

def loadDataSet():
    postingList=[['my','dog','has','flea',\
                  'problems','help','please'],
                 ['maybe','not','take','him',\
                  'to','dog','park','stupid'],
                 ['my','dalmation','is','so','cute',
                  'I','love','him'],
                 ['stop','posting','stupid','worthless','garbage'],
                 ['my','licks','ate','my','steak','how',\
                  'to','stop','him'],
                 ['quit','buying','worthless','dog','food','stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

def trainProcess(trainData, trainDataLabel):

    #print("Training data Label", trainDataLabel, sep="\n")
    #print("Training data : ", trainData, sep="\n")
    numTrainData=len(trainData)
    numberOfWords=len(trainData[0])

    numPositive = 0
    for i in trainDataLabel:
        numPositive += i

    print(numPositive)
    print(sum(trainDataLabel))
    # initialize two vectors of predicts
    p_init0 = zeros(numberOfWords)
    p_init1 = zeros(numberOfWords)
    p0sum = 0.0
    p1sum = 0.0

    print(range(numTrainData))
    for i in range(numTrainData):
        if trainDataLabel[i] == 1:
            p_init1 += trainData[i]
            p1sum += sum(trainData[i])
        else:
            p_init0 += trainData[i]
            p0sum += sum(trainData[i])

    print(p_init1/p1sum)
    print(p_init0/p0sum)
    print(sum(trainDataLabel)/float(numTrainData))
    return  p_init0/p0sum, p_init1/p1sum, sum(trainDataLabel)/float(numTrainData)


# convert word to vector for further processing
def setOfWords2Vec(wordSet,inputSet):
    # intial all value to Zero
    result = [0] * len(wordSet)
    for word in inputSet:
        if word in wordSet:
            # if appeared mark it as 1
            result[wordSet.index(word)] = 1


    return result

# get the vector name of the entity
def getNamedEntity(text):
     # preprocessing the given text
     textChunk = ne_chunk(pos_tag(word_tokenize(text)))
     continuous_chunk = []
     current_chunk = []
     # traverse the textChunk nested tree
     for i in textChunk:
             if type(i) == Tree:
                     current_chunk.append(" ".join([token for token, pos in i.leaves()]))
             elif current_chunk:
                     named_entity = " ".join(current_chunk)
                     if named_entity not in continuous_chunk:
                             continuous_chunk.append(named_entity)
                             current_chunk = []
     return continuous_chunk


#unclassified waited to be classifed
#p(wi|c0)
#p(wi|c1)
#pClass1, portion that the data has been classfied as class1
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 根据朴素贝叶斯分类函数分别计算待分类文档属于类1和类0的概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def classfySent():

    listOPosts,listClasses = loadDataSet()
    myVocabList=createVocabList(listOPosts)
    trainMat=[]

    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    print(trainMat)


    p0V,p1V,pAb = trainProcess(array(trainMat),array(listClasses))

    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry1=['stupid','garbage']
    thisDoc1=array(setOfWords2Vec(myVocabList,testEntry1))
    print(testEntry1,'classified as:',classifyNB(thisDoc1,p0V,p1V,pAb))

if __name__ == '__main__':
    sent = "BREAKING: shark are victorious in today's judgement against shark culling in the #GreatBarrierReef! Following a landmark HSI court case, a QLD tribunal has ordered the lethal component of the shark control program in the Reef must end. Full details: "
    #termWeighting(sent)
    textChunk = word_tokenize(sent)
    classfySent()
    #print(wordsToVec(textChunk, keyWords))
    #print(getNamedEntity(sent))