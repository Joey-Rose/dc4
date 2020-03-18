import pandas as pd
import gensim
import gensim.downloader as api
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from scipy import spatial
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import spacy

nlp = spacy.load('en_core_web_sm')
model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary = True, limit=500000)

def getCosineSimilarity(avgEmbedding1, avgEmbedding2):
    return 1 - spatial.distance.cosine(avgEmbedding1, avgEmbedding2)

def getDocVector(document):
    vector = []

    wordSet = word_tokenize(document)

    for word in wordSet:
        if word in model.vocab:
            #print(word)
            embeddings = model[word]
            avgEmbedding = sum(embeddings) / len(embeddings)
            vector.append(avgEmbedding)
    #print(vector)
    return sum(vector) / len(vector)

def findCosineThreshold(x_train, y_train):
    cosineSumTotal = 0
    numDuplicates = 0
    for i in range(len(x_train)):
        try:
            if y_train[i] == 0:
                continue
            #calculate word embeddings of all words in each question w/ word2vec
            avgEmbedding1 = getDocVector([x_train[i][0]])
            
            avgEmbedding2 = getDocVector(x_train[i][1])
            
            numDuplicates += 1
            #now calculate cosine similarity between vectors
            cosineSimilarity = getCosineSimilarity(avgEmbedding1, avgEmbedding2)
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
            avgEmbedding1 = getDocVector(x_test[i][0].lower())
            
            avgEmbedding2 = getDocVector(x_test[i][1].lower())
            # print(avgEmbedding1)
            # print(avgEmbedding2)
            
            #now calculate cosine similarity between vectors
            cosineSimilarity = getCosineSimilarity(avgEmbedding1, avgEmbedding2)
            
            q1_doc = nlp(x_test[i][0].lower())
            q1_tags = []
            for token in q1_doc:
                #print (token.text, token.tag_, token.head.text, token.dep_)
                q1_tags.append([token.dep_, token.text, token.head.text])
            
            q2_doc = nlp(x_test[i][1].lower())
            q2_tags = []
            for token in q2_doc:
                #print (token.text, token.tag_, token.head.text, token.dep_)
                q2_tags.append([token.dep_, token.text, token.head.text])

            amodCounter1 = 0
            amodCounter2 = 0

            if cosineSimilarity >= .7296505395554921:
                isDuplicate = 1
                for tag1 in q1_tags:
                    if tag1[0] == 'amod':
                        amodCounter1 += 1
                    for tag2 in q2_tags:
                        if tag1[0] == 'nsubj' and tag2[0] == 'nsubj':
                            string1 = tag1[1].lower() + ' ' + tag1[2].lower()
                            string2 = tag2[1].lower() + ' ' + tag2[2].lower()
                            if getCosineSimilarity(string1, string2) < .72: ##############dont know good threshold value atm!!!!!!!!
                                isDuplicate = 0 #say its not a duplicate, and move on to next question pair
                                break
                        if tag2[0] == 'amod':
                            amodCounter2 += 1
                    break            
                if amodCounter1 != amodCounter2:
                    isDuplicate = 0 #say its not a duplicate, and move on to next question pair

                test_pred.append(isDuplicate)
            else:
                test_pred.append(0)
        except:
            test_pred.append(0)
            # print(x_test[i][0])
            # print(x_test[i][1])
            # print()

    return test_pred

data = pd.read_csv('quora_duplicate_questions.tsv', sep = '\t')
#remove unimportant stuff and combine the questions into a list so you can split into test and training
data = data.drop(["id", "qid1", "qid2"], axis = 1)

data['combined']= data[['question1', 'question2']].values.tolist()
data = data.drop(['question1', 'question2'], axis = 1)
#print(data)
y = data['is_duplicate'].tolist()
x = data['combined'].to_list()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# cosineTheshold = findCosineThreshold(x_train, y_train)
# print(cosineTheshold)

test_pred = predict(x_test) 

print('Accuracy score: ', accuracy_score(y_test, test_pred))
print('Precision score: ', precision_score(y_test, test_pred))
print('Recall score: ', recall_score(y_test, test_pred))
print('F-1 score: ', f1_score(y_test, test_pred))
print(classification_report(y_test, test_pred))

# x_test = [['car has'.lower(), 'is'.lower()]]
# print(predict(x_test))

#sex one and physics one got distance of 0, but what does manipulation means got distance of 2????
#mean got avg embedding of -0.004502741495768229 while means got 0.00034679412841796874
#it could be thinking of mean as the emotion?

#print(model.similarity('mean', 'means'))
#print(model.most_similar('means')[:5])








