# Import necessary libraries
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection 
from sklearn import linear_model
from sklearn import metrics 

# Read the data from the CSV file using pandas data frame
df= pd.read_csv(r"C:\\Users\lavin\Downloads\IPL2022Batters.csv")
print(df)
# Check for null values in the entire dataset
if df.isnull().sum().sum()==0:
    print("there are 0 null values")
else:
    print("there are null values present")

#Plot graphs to explore correlations
# Total runs vs. number of 4s hit
x=df['Runs']
y=df['4s']
plt.scatter(x,y)
plt.xlabel('Total Runs')
plt.ylabel('Number of 4s Hit')
plt.title('Correlation: Total Runs vs. Number of 4s Hit')
plt.show()

#Strike rate vs. number of 4s hit
x=df['SR']
y=df['4s']
plt.scatter(x,y)
plt.xlabel('Strike Rate')
plt.ylabel('Number of 4s Hit')
plt.title('Correlation: Strike Rate vs. Number of 4s Hit')
plt.show()


# linear regression in 4s
model = linear_model.LinearRegression()
model.fit(df[['Runs']],df['4s'])
plt.ylabel('4s')
plt.xlabel('total runs')
plt.scatter(df.Runs,df['4s'],color='green')
plt.plot(df.Runs,model.predict(df[['Runs']]),color='red')
plt.show()

#features
feature=df[['Runs','Inns','Mat']]
print(feature)

#target
target=df[['4s']]
print(target)

#training and testing number of 4s
X_train, X_test, y_train, y_test =model_selection.train_test_split(feature,target,test_size=0.3)
reg=linear_model.LinearRegression()
reg.fit(X_train,y_train)

y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)

#mean squared error calculation
mse_train = metrics.mean_squared_error(y_train, y_train_pred)
mse_test = metrics.mean_squared_error(y_test, y_test_pred)
print("mse on train data",mse_train)
print("mse on test data",mse_test)
reg.fit(feature,target)




