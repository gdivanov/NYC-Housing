# NYC-Housing Predictions
Visualizing and modeling the correlation between features in New York Department of Finance data set.

We want to test a few models in our analysis - namely Random Forest, Linear/Lasso/Ridge regression types.
Note that I include an explanation for Random Forest importance biases in the Scikit-Learn library provided by http://explained.ai/rf-importance/index.html and its packages.

## I) Module Importing

First we import the modules in Python we want to use for the analysis and the modelling.

We're mostly interested in Pandas for ease of dataframe manipulation, Numpy for matrix analysis and correlation, Matplotlib for visualization, and Scikit-Learn for our supervised learning tools.

```
#Importing modules and packages
import pandas as pd
from pandas.tools.plotting import scatter_matrix 
from scipy.stats import gaussian_kde
import numpy as np
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
import matplotlib.style as style

#Modeling tool imports
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
```

## II) Data Cleaning & Inspection
Next we want to upload our data set which can be found at: https://www1.nyc.gov/site/finance/taxes/property-rolling-sales-data.page and then inspect the types that we have. Also, we need to drop some values that are either outliers or 0/null-types.

```
#Upload data set
data = pd.read_csv('C:\\...\\Python Scripts\\NYC Housing\\nyc-rolling-sales.csv')
```
Since I'm going to type the words 'SALE PRICE' along with the other features of interest in our set **a lot** I'm going to use new names as placeholders to make it faster to do dataframe cleaning.

```
#Remaning columns for convenience
price = 'SALE PRICE'
gross = 'GROSS SQUARE FEET'
land = 'LAND SQUARE FEET'
location = 'BOROUGH'
date = 'SALE DATE'
built = 'YEAR BUILT'
age = 'AGE OF BUILDING'
residential = 'RESIDENTIAL UNITS'
build_class = 'BUILDING CLASS CATEGORY'
```
The data collector mentions that they've listed the city locations in BOROUGH as 1 = Manhattan, 2 = Bronx, 3 = Brooklyn, 4 = Queens, and 5 = Staten Island. So we rename these values inside the dataframe to match that.
```
#Matching location data in BOROUGH to numerical values
data[location] = data[location].replace({1: 'Manhattan', 2: 'Bronx', 3: 'Brooklyn', 4: 'Queens', 5: 'Staten Island'})
```

Doing some inspection to the data already shows that it came pre-processed as mostly categorical object-type stuff - which we can't use because we can't do mathematical operations on it. 
&nbsp;
<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/datainfo1.jpg" width="445" height="415">
</p>
&nbsp;
We change all objects to numerical values for analysis.

```
#Change objects to numerical values
data[price] = pd.to_numeric(data[price], errors='coerce')
data[gross] = pd.to_numeric(data[gross], errors='coerce')
data[land]= pd.to_numeric(data[land], errors='coerce')
data[date] = pd.to_datetime(data[date], errors='coerce')
```
&nbsp;
<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/datainfo2.jpg" width="445" height="415">
</p>
&nbsp;
```
#Drop all duplicates except the last entry
data = data.drop_duplicates(data.columns, keep='last')

#Drop all null cases for our most important features; SALE PRICE, LAND SQUARE FEET AND GROSS SQUARE FEET
data = data[data[land].notnull()] 
data = data[data[gross].notnull()] 
data = data[data[price].notnull()]

#Drop all SALE PRICE outlier values outside a $100,000 to $4,000,000 range for better data fit in visualization
data = data[(data[price] > 100000) & (data[price] < 4000000)]

#Drop all 0 YEAR BUILT values
data = data[data[built] > 0]

#Create AGE OF BUILDING to make visualization more meaningful
data[age] = 2017 - data[built]

#Dropping 0 TOTAL UNITS values and deleting outliers in TOTAL UNITS that are above 60
data = data[(data['TOTAL UNITS'] > 0) & (data['TOTAL UNITS'] < 60)] 

#Dropping data where TOTAL NUTS are not a sum of either COMMERCIAL or RESIDENTIAL
data = data[data['TOTAL UNITS'] == data['COMMERCIAL UNITS'] + data['RESIDENTIAL UNITS']]
```

## III) Data Visualization

We want to make use of a number of various plots to inspect the distributions between features in our data set.

Some of the more interesting plots we want to use are histograms, scatterplots, boxplots, and density plots. To be able to get an understanding of how the raw data vs the cleaned data looks, I've plotted 'before and after pictures' for educational and entertainment purposes. Since the code written for both plots is the same I will refrain from writing it twice but show the raw version first and cleaned version second.

```
#Plot SALE PRICE Histogram for total range
plt.figure()
sns.set(font_scale=3.1)
hist = sns.distplot(data[price])
plt.title('Histogram of Sale Price')
plt.xlabel('Sale Price (USD)')
plt.ylabel('Normalized Frequency')
plt.savefig('fig1.pdf')
```

```
#Plotting ScatterPlot for each SALE PRICE distribution
plt.figure()
sns.set(font_scale=3.1)
g = sns.lmplot(data=data, x=price, y=gross, hue=location, fit_reg=False, size = 10.5, scatter_kws={'s':135}, legend_out = False)
for ax in g.axes.flat:
    plt.setp(ax.get_legend().get_texts(), fontsize='22') # legend text
    plt.setp(ax.get_legend().get_title(), fontsize='27') # legend title
g.despine(left=True)
plt.title('Distribution of Sale Price')
g.set(xlabel='Sale Price (USD)')
g.set(ylabel='Gross Square Feet')
plt.savefig('fig2.pdf')
```
```
#Plotting BoxPlot for SALE PRICE
plt.figure()
sns.set(font_scale=3.1)
box = sns.boxplot(x=price, data=data, linewidth=5, fliersize=12)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of Sale Price')
box.set(xlabel='Sale Price (US)')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig('fig3.pdf')

#Plotting PairPlot for SALE PRICE on GROSS SQUARE FEET and LAND SQUARE FEET
plt.figure()
sns.set(font_scale=2.3)
pair = sns.pairplot(data, vars=[price, gross, land], size=5, hue=location, aspect=1.2)
plt.savefig('fig4.pdf')

#Plot Kernal Density Plot to check population density of SALE PRICE
plt.figure()
sns.set(font_scale=3.1)
kde = sns.kdeplot(np.array(data[price]), bw=0.5)
plt.title('Kernal Density of Sale Price')
kde.set(xlabel='Sale Price (USD)')
kde.set(ylabel='Unit Probability')
plt.savefig('fig5.pdf')

#Compute the correlation matrix
sns.set(font_scale=2.1)
d = data[['RESIDENTIAL UNITS', 'COMMERCIAL UNITS', 'TOTAL UNITS','GROSS SQUARE FEET','SALE PRICE', 'AGE OF BUILDING', 'LAND SQUARE FEET']]
corr = d.corr()

#Generate entries of zeros for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

#Setting the figure
f, ax = plt.subplots(figsize=(70, 16))

#Generate diverging colormap
cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

#Heatmap with the zeros and correct aspect ratio
sns.heatmap(corr, mask=mask, square=True, linewidths=.5, annot=True, cmap=cmap)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
plt.title('Correlation Matrix for New York Housing Variables 2017-2018')
plt.tight_layout()
plt.show()

#Plot regression for square feet and price
sns.regplot(x=gross, y=price, data=data, fit_reg=False, scatter_kws={'alpha':0.3}, logistic=True)
plt.title('Gross Square Feet vs Sale Price')
plt.show()

#Plot regression for age and price
sns.regplot(x=age, y=price, data=data, fit_reg=False, scatter_kws={'alpha':0.1}, logistic=True)
plt.title('Sale Price Distribution by Building Age')
plt.show()

#Plot location
sns.boxplot(x=location, y=price, data=data)
plt.title('Sale Price Distribution by Location')
plt.show()
```

## IV) Tranforming Data for Model Building
### Standardization of Varying Numerical Data

We're almost at the 'cool' part.

Because we are utilizing certain regression models in training our data - Linear, Lasso, and Ridge - and because our data is of differing varieties and magnitudes - Price (USD), Square Footage, Units, Age - we need to be working with standardized data. This means we need to center our price distribution (dependant variable) along a zero mean and transform all the data to a logarithmic scale.

This will help our ML tools in Scikit-Learn perform better because if we've scaled our features our models will converge faster - at least that's the hope.

However, out of interest we won't be standardizing our Logistic Regression model.

One thing to note is that scaling data for logarithmic data means that we can't have any zeroes in the set because log(0) is undefined. Instead we can treat those zeroes in the original numerical space as zeroes in logarithm space by setting all zero values to 1 since log(1) = 0. So then we have to add 1 to all the columns that have minimum values of 0 or else we're going to get errors.

The minimum column Sale Price value is not zero so we can skip adding a 1 to it as it won't make a difference in our scaling.

### i) Preparing the Standardization

```
#Add 1 to all numerical data other than SALE PRICE
data[commercial] = data[commercial] + 1
data[residential] = data[residential] + 1
data[gross] = data[gross] + 1
data[land] = data[land] + 1
data[age] = data[age] + 1

#Standardizing and transforming into logarithmic scale
data[price] = StandardScaler().fit_transform(np.log(data[price]).reshape(-1,1))
data[commercial] = StandardScaler().fit_transform(np.log(data[commercial]).reshape(-1,1))
data[residential] = StandardScaler().fit_transform(np.log(data[residential]).reshape(-1,1))
data[gross] = StandardScaler().fit_transform(np.log(data[gross]).reshape(-1,1))
data[land] = StandardScaler().fit_transform(np.log(data[land]).reshape(-1,1))
data[age] = StandardScaler().fit_transform(np.log(data[age]).reshape(-1,1))

#Plot the new Standardized SALE PRICE
sns.distplot(data[price])
plt.title('Standardized Histogram of Sale Price')
plt.show()

#Standardizing and transforming into logarithmic scale
data[price] = StandardScaler().fit_transform(np.log(data[price]).reshape(-1,1))

#Plot the new Standardized SALE PRICE
sns.distplot(data[price])
plt.title('Standardized Histogram of Sale Price')
plt.show()
```

### One Hot Encoding for Categorical Data

We also want to include the categorical data in our model - Location, Building Class Category - so we need to interpret to the machine in some way. But the machine can't tell what 'Bronx" or what a 'RENTAL' is so we need to tell it what it is. One of the ways we can do this is by interpreting categories numerically; something the machine **can** read.

For example we can show how this works by looking at 2 Locations and 2 Building Class Categories in a vector like so:

```
["Bronx", "Brooklyn", "ONE FAMILY DWELLINGS", "OFFICE BUILDING"]
```
The one hot encoding transforms the categories as integers 1, 2, 3, 4, and then maps them to a binary vector of a 1 or a 0, say:
```
[0, 1, 1, 0]
```
This would translate to a one family home in Brooklyn.

### ii) Preparing the One Hot Encoding

We know there are 5 unique locations and 31 building class categories. Therefore, we take the categorical data into a new vector and begin encoding using Pandas' get_dummies operator which transform categorical data into indicators and into a new dataframe.

```
#Standardizing and transforming into logarithmic scale
data[price] = StandardScaler().fit_transform(np.log(data_model[price]).reshape(-1,1))

#Plot the new Standardized SALE PRICE
sns.distplot(data[price])
plt.title('Standardized Histogram of Sale Price')
plt.show()

#Splitting categorical data
one_hot_data = [location, build_class]

#Encoding and updating the data
one_hot_encoded = pd.get_dummies(data[one_hot_data])
one_hot_encoded.info(verbose=True, null_counts=True)
data = data.drop(one_hot_data, axis=1)
data = pd.concat([data, one_hot_encoded], axis=1)

#Standardizing and transforming into logarithmic scale
data[price] = StandardScaler().fit_transform(np.log(data_model[price]).reshape(-1,1))

#Plot the new Standardized SALE PRICE
sns.distplot(data[price])
plt.title('Standardized Histogram of Sale Price')
plt.show()
```
## V) Training & Modelling the Data

### Train-Test Splitting

So we've arrived at our final destination - training and testing the models. 

Dear god, yes.

Our data is now ready to be split into their respective training and testing populations. A general rule of thumb is that 80/20 is a good starting point so we'll go with that for now.

Also, I'm really only interested in passing through specific features into our models for predictions.

```
#Splitting the variables out that we are interested in modeling
columns = [location, commercial, residential, gross, land, age, build_class, price]
data_mod = data.loc[:,columns]
```
Now we have the data we want so let's split it into training and validation samples.
```
#Split the data
training, testing = train_test_split(data_mod, test_size=0.2, random_state=0)
print("Total Data Set = %i; Training Set = %i, Testing Set = %i"\
     %(data.shape[0],training.shape[0],testing.shape[0]))
```
Okay, we have a decently sized pool. We could also test 75/25 or any variant around 80-70/30-20 but we'll save that for another project.

We're now ready to choose the SALE PRICE as our target set (y) and the rest of the features as our training set (X). We also want to choose our validation set in the same way.
```
#Choosing training set
df_train = training.loc[:,data_mod.columns]
X_train = df_train.drop([price], axis=1)
y_train = df_train.loc[:, [price]]

#Choosing validation set
df_test = testing.loc[:,data_mod.columns]
X_test = df_test.drop([price], axis=1)
y_test = df_test.loc[:, [price]]
```
### Modelling & Choosing the "k" in K-Fold 

Before we begin running our data through we need to choose a k value for a cross-validation. Having a higher k value means that more iterations will take place at a lower sample size. So it's a trade-off.

On one hand if we have a higher k value then we have a higher computational time, higher variance, and lower bias. If we have a smaller k value then we have a lower computational time, lower variance, and higher bias.

We choose k = 5 because we only have ~26,000 data points. However, you should always note how much data you have to work with. Since we deleted a lot of our data we aren't left with a whole lot - so choosing k = 5 seems reasonable.

### Linear Regression Model

It's very __'fitting'__ if we begin with the most common regression model - Linear Regression. 

```
#Fit Linear Regressor to training data
linreg = LinearRegression()
linreg.fit(X_train, y_train)

#Predict SALE PRICE labels and compute 5-Fold Cross-Validation
y_pred = linreg.predict(X_test)
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=5)
```
Computing our coefficient of determination (R^2), Root Mean Square Error, and Mean of the CV-Score we get:

```
print("R^2: {}".format(linreg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error: {}".format(rmse))

print("Mean 5-Fold CV Score: {}".format(np.mean(linreg_cv)))
# Print the 5-fold cross-validation scores
print(linreg_cv)
```

### Ridge Regression Model

For Ridge Regression we must choose a suitible value for our regularization parameter 'alpha'. This value allows us to scale the coefficients used to smoothen our model out. However, we need to find these values to see which one will fit our coefficients well.

We will set up an array of possible values which will begin very large and end up very small. This will let us choose a value that will give us a the smallest cross-validation error between training sets that we can find.

```
#Create array of alpha values
alphas = 10**np.linspace(10,-2,100)*0.5

#Create Ridge Regressor
ridge = Ridge()

#Create X and y to find coefficients for total data points
X = data_mod.drop([price], axis=1)
y = data_mod.loc[:, price]

#Store coefficients into vector
coefs = []
for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
np.shape(coefs)

#Plot regularization parameters for Ridge
ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

#Calculate best alpha with smallest cross-validation error for Ridge
ridgecv = RidgeCV(alphas=alphas, scoring='mean_squared_error', normalize=True)
ridgecv.fit(X_train, y_train)
ridgecv.alpha_

#Update Ridge regressor using best alpha
ridge = Ridge(alpha=ridgecv.alpha_, normalize=True)

#Fit the model
ridge.fit(X_train, y_train)

#Predict
y_pred_ridge = ridge.predict(X_test)

#Perform 5-fold cross-validation for scoring
ridge_cv = cross_val_score(ridge, X_train, y_train, cv=5)
print("R^2: {}".format(ridge.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
print("Root Mean Squared Error: {}".format(rmse))

#Print the 5-fold cross-validation scores
print("Average 5-Fold CV Score: {}".format(np.mean(ridge_cv)))
print(ridge_cv)

#Show Ridge coefficients
ridge.fit(X,y)
pd.Series(ridge.coef_, index=X.columns)
```
### Lasso Regression Model

Just like Ridge Regression, Lasso tries to smoothen out our model using regularization. However, Lasso may allow smaller contributions that are negligible to be zeroed out completely.

Let's check the results.

```
#Create Lasso regressor with maximum iterations set to 10,000
lasso = Lasso(max_iter=10000)

#Store coefficients into vector
coefs = []
for a in alphas:
    lasso.set_params(alpha=a)
    lasso.fit(scale(X_train), y_train)
    coefs.append(lasso.coef_)
    
#Plot regularization parameters for Lasso
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

#Calculate best alpha with smallest cross-validation error for Lasso
lassocv = LassoCV(alphas=None, max_iter=100000, cv=5)
lassocv.fit(X_train, y_train)

#Set Lasso regularization parameters to data
lasso.set_params(alpha=lassocv.alpha_)
lasso.fit(X_train, y_train)

#Predict
y_pred_lasso = lasso.predict(X_test)

#Perform 5-fold cross-validation for scoring
lasso_cv = cross_val_score(lasso, X_train, y_train, cv=5)
print("R^2: {}".format(lasso.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
print("Root Mean Squared Error: {}".format(rmse))

#Print the 5-fold cross-validation scores
print("Average 5-Fold CV Score: {}".format(np.mean(lasso_cv)))
print(lasso_cv)

#Show Lasso coefficients
pd.Series(lasso.coef_, index=X.columns)
```
### Random Forest Model

Last but not least we want to use one decision tree type of model to make some regressive predictions as well.

Random forest will branch subsets of our data into independant, smaller decision trees, and then boost out a weighted average of what features are most important. This is the so-called 'importance' determination of features - which is described in a more thorough investigation in another text file.

First we create the model.

```
#Create Random Forest Regressor
rf_reg = RandomForestRegressor()

#Fit the regressor
rf_reg.fit(X_train, y_train)

#Predict
y_pred_rf = rf_reg.predict(X_test)

#Compute 5-fold cross-validation for scoring
rf_cv = cross_val_score(rf_reg, X_train, y_train, cv=5)
print("R^2: {}".format(rf_reg.score(X_test, y_test)))
rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf))
print("Root Mean Squared Error: {}".format(rmse))

#Print the 5-fold cross-validation scores
print("Average 5-Fold CV Score: {}".format(np.mean(rf_cv)))
#Print the 5-fold cross-validation scores
print(rf_cv)
```
We now want to showcase the importance that has been measured by our model. We create a new dataframe with this list in descending order to see how valuable each feature actually is.

```
importance = pd.DataFrame(list(zip(X_train.columns, np.transpose(rf_reg.feature_importances_)))).sort_values(1, ascending=False)
print(importance)
```

