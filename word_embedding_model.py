import pandas as pd
import gensim
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

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
        if y_train[i] == '1':
            continue
        numDuplicates += 1
        #calculate word embeddings of all words in each question w/ word2vec
        avgEmbedding1 = getDocVector(x_train[i][0])
        
        avgEmbedding2 = getDocVector(x_train[i][1])

        #now calculate cosine similarity between vectors
        cosineSimilarity = 99 ##FIXME make actual value
        cosineSumTotal += cosineSimilarity

    return cosineSumTotal / numDuplicates

model = api.load("glove-twitter-25")

data = pd.read_csv('quora_duplicate_questions.tsv', sep = '\t')
#remove unimportant stuff and combine the questions into a list so you can split into test and training
data = data.drop(["id", "qid1", "qid2"], axis = 1)

data['combined']= data[['question1', 'question2']].values.tolist()
data = data.drop(['question1', 'question2'], axis = 1)
#print(data)
y = data['is_duplicate'].tolist()
x = data['combined'].to_list()

x_train, x_test, y_train, y_test = train_test_split(x, y)

cosineTheshold = findCosineThreshold(x_train, y_train)








