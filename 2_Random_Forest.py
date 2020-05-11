#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from scipy.special import comb

#载入红酒数据集
wine = load_wine()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(wine.data, wine.target, test_size=0.3, random_state=1)

#决策树，随机森林效果对比
clf = DecisionTreeClassifier(random_state=0)
rfc = RandomForestClassifier(random_state=0)

clf = clf.fit(Xtrain, Ytrain)
rfc = rfc.fit(Xtrain, Ytrain)

score_c = clf.score(Xtest, Ytest)
score_r = rfc.score(Xtest, Ytest)

print("Decision_Tree:{}".format(score_c), "Random_Forest:{}".format(score_r))#-----输出效果对比


rfc = RandomForestClassifier(n_estimators=25)
rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10)#一个数组包含了每一次交叉验证的结果

clf = DecisionTreeClassifier()
clf_s = cross_val_score(clf, wine.data, wine.target, cv=10)

plt.plot(range(1, 11), rfc_s, label="RandomForest")
plt.plot(range(1, 11), clf_s, label="Decision Tree")
plt.legend()
plt.show()

#进行十次十折交叉验证，每循环一次都取均值
rfc_l = []
clf_l = []
for i in range(10):
    rfc = RandomForestClassifier(n_estimators=25)
    rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
    rfc_l.append(rfc_s)
    clf = DecisionTreeClassifier()
    clf_s = cross_val_score(clf, wine.data, wine.target, cv=10).mean()
    clf_l.append(clf_s)

plt.plot(range(1, 11), rfc_l, label="Random Forest")
plt.plot(range(1, 11), clf_l, label="Decision Tree")
plt.title('Crossvalidation_10times')
plt.legend()
plt.show()



superparameters = []
for i in range(10):
    rfc = RandomForestClassifier(n_estimators=i + 1)
    rfc_s = cross_val_score(rfc, wine.data, wine.target, cv=10).mean()
    superparameters.append(rfc_s)
print(max(superparameters), superparameters.index(max(superparameters)) + 1)
plt.figure(figsize=[20, 5])
plt.plot(range(1, 11), superparameters)
plt.title('Val_Score with different n_estimators')
plt.show()

#想得出更高的分数可以使用网格搜索来细化参数，在此就不进行演示了


rfc = RandomForestClassifier(n_estimators=20, random_state=2)
rfc = rfc.fit(Xtrain, Ytrain)

# 随机森林的重要属性之一：estimators，查看森林中树的状况
print(rfc.estimators_)
for i in range(len(rfc.estimators_)):
    print(rfc.estimators_[i].random_state)
print('*'*100)

#bagging
# 无需划分训练集和测试集
rfc = RandomForestClassifier(n_estimators=25, oob_score=True)  # 默认为False
rfc = rfc.fit(wine.data, wine.target)

# 重要属性oob_score_,得出分数
print(rfc.oob_score_)
print('-'*100,'\r\n')

rfc = RandomForestClassifier(n_estimators=25)
rfc = rfc.fit(Xtrain, Ytrain)
print(rfc.score(Xtest, Ytest))
print(rfc.apply(Xtest),'This is the index')# apply返回每个测试样本所在的叶子节点的索引
print(rfc.predict(Xtest),'This is the prediction','\r\n')  # predict返回每个测试样本的分类/回归结果
print(rfc.predict_proba(Xtest),'This is the probability')

x = np.linspace(0, 1, 20)
y = []
for epsilon in np.linspace(0, 1, 20):
    E = np.array([comb(25, i) * (epsilon ** i) * ((1 - epsilon) ** (25 - i)) for i in range(13, 26)]).sum()
    y.append(E)
plt.plot(x, y, "o-", label="when estimators are different")
plt.plot(x, x, "--", color="red", label="if all estimators are same")
plt.xlabel("individual estimator's error")
plt.ylabel("RandomForest's error")
plt.legend()
plt.show()
