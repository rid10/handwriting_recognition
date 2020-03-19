import warnings
warnings.filterwarnings(action="ignore")

#Standard scientific Python imports
import matplotlib.pyplot as plt

#Import datasets, classifiers, and performance metrics
from sklearn import datasets, svm

#The digits dataset
digits=datasets.load_digits()

#print("digits:",digits.keys())
print("digits.target---- : ",digits.target)
images_and_labels=list(zip(digits.images,digits.target))
print("len(images_and_labels",len(images_and_labels))

for index, [image,label] in enumerate(images_and_labels[:5]):
    print("index:",index,"image:\n",image,"label: ",label)
    plt.subplot(2,5,index+1) #Position numbering starts from 1
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Training: %i' % label)

#plt.show()


#To apply a classifier on this data we need to flatten the image to turn the data in a (sample, feature) matrix:
n_samples=len(digits.images)
print("n_samples: ",n_samples)

imageData=digits.images.reshape((n_samples,-1))

print("After Reshaped: len(imagesData[0]): ",len(imageData[0]))

#Create a classifier: a support vector classifier
classifier=svm.SVC(gamma=0.001)

#We learn the digits on the first half of the digits
classifier.fit(imageData[ : n_samples//2],digits.target[:n_samples//2])

#Now predict the value of the digit on the second half
originalY=digits.target[n_samples//2:]
predictedY=classifier.predict(imageData[n_samples//2:])

images_and_predictions = list(zip(digits.images[n_samples//2:],predictedY))

for index, [image,prediction] in enumerate(images_and_predictions[:5]):
    plt.subplot(2,5,index+6)
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Prediction: %i'% prediction)

print("Original Values: ",digits.target[n_samples//2: (n_samples//2)+5])
plt.show()

#Install Pillow library
#from PIL
#from scipy.misc import imread , imresize, bytescale
from sklearn.externals._pilutil import imresize, imread, bytescale
img = imread("three.jpeg")
img=  imresize(img,(8,8))

classifier=svm.SVC(gamma=0.001)
classifier.fit(imageData[:],digits.target[:])

img=img.astype(digits.images.dtype)
img=bytescale(img,high=16.0,low=0)

print("img.shape : ",img.shape)
print("\n",img)

x_testData=[]
for row in img:
    for col in row:
        x_testData.append(sum(col)/3.0)
print("x_testData: \n",x_testData)
print("len(x_testData):",len(x_testData))

x_testData=[x_testData]
print("len(x_testData) : ",len(x_testData))

result = classifier.predict(x_testData)
print("Machine Output = ",result)

plt.show()
