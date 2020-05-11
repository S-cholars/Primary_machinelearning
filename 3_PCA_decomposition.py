#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA

faces = fetch_lfw_people(min_faces_per_person=60)#实例化   min_faces_per_person=60：每个人取出60张
print(faces)#字典形式

print(faces.images.shape)  #第一个是矩阵中图像的个数，62是每个图像的特征矩阵的行，47是每个图像的特征矩阵的列
print(faces.data.shape)  #行是样本，列是样本相关的所有特征：2914 = 62 * 47

X = faces.data
print(X.shape)
#创建画布，先绘制原本数据 的人脸
fig,axes=plt.subplots(5,5,subplot_kw=dict(xticks=[],yticks=[]))
for i,ax in enumerate(axes.flat):
    ax.imshow(faces.images[i,:,:],cmap='binary_r')
plt.show()

#接下来使用PCA降维
# pca=PCA(n_components='mle').fit(X) 使用不成功，因为样本数量没有大于特征数量
pca=PCA(n_components=0.98,svd_solver='full').fit(X)
v=pca.components_
print(v.shape)
fig,axes=plt.subplots(5,5,subplot_kw=dict(xticks=[],yticks=[]))
for i,ax in enumerate(axes.flat):
    ax.imshow(v[i,:].reshape(62,47),cmap='binary_r')
    # ax.imshow(v.reshape(62, 47)[i,:,:], cmap='binary_r'),注意不能写成这样，取索引必须在前，然后才能reshape
plt.show()

#实例化
pca = PCA(150)
#拟合+提取结果
X_dr = pca.fit_transform(X)  
X_inverse = pca.inverse_transform(X_dr)
print(X_inverse.shape#)
fig,axes=plt.subplots(2,10,subplot_kw=dict(xticks=[],yticks=[]))
for i in range(10):
      ax[0,i].imshow(faces.images[i,:,:],cmap='gray')
      ax[1,i].imshow(X_inverse[i].reshape(62,47),cmap='gray')
 plt.show()











