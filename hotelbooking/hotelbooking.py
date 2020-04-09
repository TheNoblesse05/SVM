'''
Task Details -
Can we predict the possibility of a booking for a hotel based on the previous_cancellation as the target?
Expected Submission - 
Two columns --> Hotel Name and Booking_Possibility (0 or 1)

Author - Vedant Tilwani
'''
import numpy as np 
import pandas as pd
import seaborn as sns 						#used for plotting graphs. Can be used with matplotlib.
import matplotlib.pyplot as plt 

data = pd.read_csv('hotel_bookings.csv') 	#reads the data

print(data.head(),'\n') 					#prints the first 5 entries
print(data.tail(),'\n')					#prints the last 5 entries

print('Shape of the dataset is ',data.shape,'\n')		#prints the shape of the dataset

print(data.info(),'\n') 					shows the datatypes of the columns					

print(data.describe().T,'\n')				#prints the count, mean, max of the int and float type columns
print(data.describe(include='object').T,'\n')  #prints the count, unique, top and freq of the object type columns

print(data.isna().sum(),'\n') 			#prints the number of NULL/missing values in each column

print(data.select_dtypes(include='object').columns,'\n')	#prints all the columns that have datatype as object

data = data.drop(['agent','company','reservation_status_date'],axis=1) 		#drops the row(axis=0) or column(axis=1) from the database
print(data)

print(data['country'].mode())
data['country'] = data['country'].replace(np.nan,'PRT')						#replaces the null in country with PRT

sns.countplot(data['hotel'])				#a plot to show the frequency/count of variables in a column
plt.show()
sns.countplot(data['arrival_date_month'])
plt.show()
sns.countplot(data['is_canceled'])
plt.show()

print(data['is_canceled'].value_counts())

sns.countplot(data['meal'])
plt.show()
sns.countplot(data['market_segment'])
plt.show()
sns.countplot(data['distribution_channel'])
plt.show()
sns.countplot(data['reserved_room_type'])
plt.show()
sns.countplot(data['assigned_room_type'])
plt.show()
sns.countplot(data['deposit_type'])
plt.show()
sns.countplot(data['customer_type'])
plt.show()
sns.countplot(data['reservation_status'])
plt.show()

sns.barplot(data['reservation_status'],data['arrival_date_year'],) 		#????
plt.show()

print(data.corr())							#prints the correlation between different columns of data

sns.barplot(data['arrival_date_year'],data['previous_cancellations'])
plt.show()
sns.barplot(data['arrival_date_year'],data['previous_bookings_not_canceled'])
plt.show()
sns.barplot(data['arrival_date_month'],data['previous_cancellations'])
plt.show()
sns.barplot(data['arrival_date_month'],data['previous_bookings_not_canceled'])
plt.show()
sns.barplot(data['arrival_date_month'],data['is_canceled'])
plt.show()
sns.barplot(data['arrival_date_year'],data['is_canceled'])
plt.show()

df = pd.get_dummies(data,prefix=['hotel', 'arrival_date_month', 'meal', 'market_segment', 		#converts data in object datatype to int
       'distribution_channel', 'reserved_room_type', 'assigned_room_type',
       'deposit_type', 'customer_type', 'reservation_status',
       'reservation_status_date'])
print(df.head())

print('Shape of dataset ',df.shape,'\n')
print('Size of dataset ',df.size,'\n')

for i in df.columns: 							#prints the columns which have null
	if df[i].isnull().sum()!= 0:
		#print('{} {}'.format(i,df[i].isnull().sum()))
		df[i] = df[i].replace(np.nan,'0')


print(df.corr()) 								#correlation of the new table having all int values
sns.heatmap(df.corr())
plt.show()

from sklearn.model_selection import train_test_split

X = df.drop('is_canceled',axis=1)
Y = df['is_canceled']


# LINEAR SVM
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.3) 		#splits data into test and train
svm_clf = Pipeline((('scaler',StandardScaler()),('linear_svc',LinearSVC(C=1,loss='hinge')),)) 		#scales the data and performs SVM classification

svm_clf.fit(X_train,Y_train)				#trains the data

Y_pred = svm_clf.predict(X_test)			#predictions made my the SVM classifier

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y_test,Y_pred)
print(confusion_matrix)

from sklearn.metrics import accuracy_score 
accuracy = accuracy_score(Y_test,Y_pred)
print('Accuracy of the SVM Model is ',accuracy)
