import os
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC


def calcuAccu(predict):
    predict_target = []
    for i in range(145): predict_target.append(1)
    for i in range(500): predict_target.append(2)
    for i in range(123): predict_target.append(3)
    for i in range(500): predict_target.append(4)
    for i in range(301): predict_target.append(5)
    for i in range(354): predict_target.append(6)
    for i in range(22): predict_target.append(7)
    for i in range(500): predict_target.append(8)
    for i in range(402): predict_target.append(9)
    for i in range(103): predict_target.append(10)
    for i in range(500): predict_target.append(11)
    for i in range(184): predict_target.append(12)
    for i in range(437): predict_target.append(13)
    for i in range(469): predict_target.append(14)
    for i in range(75): predict_target.append(15)
    for i in range(500): predict_target.append(16)
    for i in range(38): predict_target.append(17)

    # print(len(predict_target) == len(predict))
    pos, neg = 0, 0
    for i in range(len(predict)):
        if predict_target[i] == predict[i]:
            pos += 1
        else:
            neg += 1
    return pos, neg, pos / (pos + neg)


predict_data = []
list = os.listdir("csv")
for i in range(0,len(list)):
    path = os.path.join("csv",list[i])
    if os.path.isfile(path):
        temp = np.loadtxt(open(path,'rb'), dtype=str, delimiter='\n',skiprows=1)
        predict_data.extend(temp)


train_data = np.loadtxt(open("train_data.csv", "rb"), dtype=str, delimiter="\n")
train_target = np.loadtxt(open("train_target.csv", "rb"), dtype=int, delimiter="\n")

vectorizer = HashingVectorizer(stop_words="english", n_features=5000, non_negative=True)
fea_train = vectorizer.fit_transform(train_data)
fea_predict = vectorizer.fit_transform(predict_data)


knnclf = KNeighborsClassifier(n_neighbors=17)
knnclf.fit(fea_train, train_target)
nbclf = MultinomialNB(alpha=0.01)
nbclf.fit(fea_train, train_target)
svclf = SVC(kernel="linear")
svclf.fit(fea_train, train_target)

# predict = nbclf.predict(fea_predict)
# np.savetxt("predict.csv", predict, fmt="%d")
# p, n, a = calcuAccu(predict)
# print("NaiveBayes:\t Pos - " + str(p) + ", Neg - " + str(n) + ", Accu - " + str(a) + ";")
#
# predict = knnclf.predict(fea_predict)
# np.savetxt("predict.csv", predict, fmt="%d")
# p, n, a = calcuAccu(predict)
# print("kNN:\t\t Pos - " + str(p) + ", Neg - " + str(n) + ", Accu - " + str(a) + ";")
#
predict = svclf.predict(fea_predict)
np.savetxt("predict.csv", predict, fmt="%d")
# p, n, a = calcuAccu(predict)
# print("SVM:\t\t Pos - " + str(p) + ", Neg - " + str(n) + ", Accu - " + str(a) + ";")

type = [0,0,0,0,0,0,0]
for t in predict:
    type[t] += 1
    type[0] += 1
print("Critical:\t\t" + str(type[1]))
print("Major:\t\t\t"+str(type[2]))
print("Normal:\t\t\t" + str(type[3]))
print("Minor:\t\t\t"+ str(type[4]))
print("Trivial:\t\t" + str(type[5]))
print("Enhancement:\t"+str(type[6]))



