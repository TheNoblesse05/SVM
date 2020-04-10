'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
-ticket Ticket Number
-fare Passenger Fare (British pound)
-cabin Cabin
-embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
-boat Lifeboat
-body Body Identification Number
-home.dest Home/Destination
'''


import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 

data = pd.read_excel('titanic.xls')

print('shape of the data is ',data.shape)
print(data.isna().sum())
print(data.select_dtypes(include='object').columns)

# sns.countplot(data['sex'])
# plt.show()
# sns.countplot(data['pclass'])
# plt.show()
# sns.countplot(data['survived'])
# plt.show()

# sns.barplot(data['sex'],data['survived'],) 	
# plt.show()
# sns.barplot(data['pclass'],data['survived'],) 	
# plt.show()

data = data.drop(['name','body','fare','ticket','cabin','boat','embarked','home.dest'],axis=1)
print('Shape of the data is ',data.shape)

df = pd.get_dummies(data,prefix=['sex'])
# print(df.corr())

df['age'] = df['age'].replace(np.nan,29.88)

from sklearn.model_selection import train_test_split

X = df.drop('survived',axis=1)
Y = df['survived']

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.25) 		#splits data into test and train
poly_kernel_svm_clf = Pipeline((("scaler", StandardScaler()),("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
)) 		#scales the data and performs SVM classification
poly_kernel_svm_clf.fit(X_train,Y_train)	
Y_pred = poly_kernel_svm_clf.predict(X_test)			#predictions made my the SVM classifier

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)

from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(Y_test,Y_pred)
print('Accuracy of the SVM Model is ',accuracy)

