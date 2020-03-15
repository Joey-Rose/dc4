import pandas as pd
import gensim
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from scipy import spatial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def getDocVector(document):
    vector = []

    wordSet = word_tokenize(document)

    for word in wordSet:
        if word in model.vocab:
            embeddings = model[word]
            avgEmbedding = sum(embeddings) / len(embeddings)
            vector.append(avgEmbedding)
    return sum(vector) / len(vector)

def findCosineThreshold(x_train, y_train):
    cosineSumTotal = 0
    numDuplicates = 0
    for i in range(len(x_train)):
        try:
            if y_train[i] == 0:
                continue
            #calculate word embeddings of all words in each question w/ word2vec
            avgEmbedding1 = getDocVector(x_train[i][0])
            
            avgEmbedding2 = getDocVector(x_train[i][1])
            
            numDuplicates += 1
            #now calculate cosine similarity between vectors
            cosineSimilarity = 1 - spatial.distance.cosine(avgEmbedding1, avgEmbedding2)
            cosineSumTotal += cosineSimilarity
        except:
            pass
            # print(x_train[i][0])
            # print(x_train[i][1])
            # print()

    return cosineSumTotal / numDuplicates

def predict(x_test):
    #1 means duplicate
    test_pred = []
    for i in range(len(x_test)):
        try:
            #calculate word embeddings of all words in each question w/ word2vec
            avgEmbedding1 = getDocVector(x_test[i][0])
            
            avgEmbedding2 = getDocVector(x_test[i][1])
            
            #now calculate cosine similarity between vectors
            cosineSimilarity = 1 - spatial.distance.cosine(avgEmbedding1, avgEmbedding2)
            if cosineSimilarity >= .7296505395554921:
                test_pred.append(1)
            else:
                test_pred.append(0)
        except:
            test_pred.append(0)
            pass
            # print(x_test[i][0])
            # print(x_test[i][1])
            # print()

    return test_pred

model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True, limit=500000)
# model = gensim.models.KeyedVectors.load_word2vec_format('path-to-vectors.txt', binary=False)


data = pd.read_csv('quora_duplicate_questions.tsv', sep = '\t')
#remove unimportant stuff and combine the questions into a list so you can split into test and training
data = data.drop(["id", "qid1", "qid2"], axis = 1)

data['combined']= data[['question1', 'question2']].values.tolist()
data = data.drop(['question1', 'question2'], axis = 1)
#print(data)
y = data['is_duplicate'].tolist()
x = data['combined'].to_list()

x_train, x_test, y_train, y_test = train_test_split(x, y)

# cosineTheshold = findCosineThreshold(x_train, y_train)
# print(cosineTheshold)

test_pred = predict(x_test) 

print('Accuracy score: ', accuracy_score(y_test, test_pred))
print('Precision score: ', precision_score(y_test, test_pred))
print('Recall score: ', recall_score(y_test, test_pred))
print('F-1 score: ', f1_score(y_test, test_pred))
print(classification_report(y_test, test_pred))








