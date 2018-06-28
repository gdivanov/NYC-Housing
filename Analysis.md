# NYC-Housing Predictions
Visualizing and modeling the correlation between features in New York Department of Finance data set and see which model should be chosen for future investigation.

We want to test 3 models in our analysis - Random Forest Decision Tree, Lasso, and Ridge regression.

Note that the next step of this analysis is to include an explanation for Random Forest Importance biases in the default Scikit-Learn library. The Random Forest case study will be made possible by http://explained.ai/rf-importance/index.html and the packages given.

## I) Module Importing

First we import the modules in Python we want to use for the analysis and the modeling.

We're mostly interested in Pandas for ease of dataframe manipulation, Numpy for matrix analysis and correlation, Matplotlib for visualization, and Scikit-Learn for our supervised learning tools.

```
#Import all dataframe, mathematical, and visualization modules
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
sns.set(style='whitegrid', context='notebook', palette='deep') 

#Import ML tools and metrics
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
```

## II) Data Cleaning & Inspection

Next we want to upload our data set which can be found at: https://www1.nyc.gov/site/finance/taxes/property-rolling-sales-data.page (or on the main page of my NYC Housing GitHub as "nyc-rolling-sales.csv".

Before we do that it's important to note that we are really interested mostly in the price, square footage, location, and the overall age of the houses and how these correlate. So we will mostly be working on cleaning and preparing these values for modeling later.

```
#Upload data set
data = pd.read_csv('C:\\...\\Python Scripts\\NYC Housing\\nyc-rolling-sales.csv')
```
Since I'm going to type the words 'SALE PRICE' along with the other features of interest in our set **a lot** I'm going to use new names as placeholders to make it faster to do dataframe cleaning.

```
#Renaming columns for convenience
price = 'SALE PRICE'
gross = 'GROSS SQUARE FEET'
land = 'LAND SQUARE FEET'
location = 'BOROUGH'
date = 'SALE DATE'
built = 'YEAR BUILT'
age = 'AGE OF BUILDING'
residential = 'RESIDENTIAL UNITS'
build_class = 'BUILDING CLASS CATEGORY'
total = 'TOTAL UNITS'
```

To see our initial starting point let's check the size of our data set:

```
data.info()
```
We see that the dataframe is `(84548, 22)`. We'll obviously need to do some cleaning, though.

The data collector mentions that they've listed the city locations in BOROUGH as numerical values rather than categorical.
```
1 = Manhattan, 2 = Bronx, 3 = Brooklyn, 4 = Queens, and 5 = Staten Island 
```

So we rename these values inside the dataframe to match that.
```
#Matching location data in BOROUGH to numerical values
data[location] = data[location].replace({1: 'Manhattan', 2: 'Bronx', 3: 'Brooklyn', 4: 'Queens', 5: 'Staten Island'})
```

Doing some inspection to the data already shows that it came pre-processed as mostly categorical object-type stuff - which we can't use because we can't do mathematical operations on it. We show this below.

&nbsp;
<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/datainfo0.jpg" width="445" height="415">
</p>
&nbsp;

So let's change all objects to numerical values for analysis.

```
#Change objects to numerical values
data[price] = pd.to_numeric(data[price], errors='coerce')
data[gross] = pd.to_numeric(data[gross], errors='coerce')
data[land]= pd.to_numeric(data[land], errors='coerce')
data[date] = pd.to_datetime(data[date], errors='coerce')
```
&nbsp;
<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/datainfo1.jpg" width="445" height="415">
</p>
&nbsp;

Next, we always want to take care of null values, duplicates, and any other logic-oriented outliers to keep only what makes sense for an analysis.

First we drop the duplicates.
```
#Drop all duplicates except the last entry
data = data.drop_duplicates(data.columns, keep='last')
```
Next we drop null values for the features of interest.
```
#Drop all null cases for our most important features; SALE PRICE, LAND SQUARE FEET AND GROSS SQUARE FEET
data = data[data[land].notnull()] 
data = data[data[gross].notnull()] 
data = data[data[price].notnull()]
```

&nbsp;
<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/datainfo2.png" width="445" height="415">
</p>
&nbsp;

Now we're in pretty good shape.

It's time to start thinking about ranges on all of our values. Does it make sense to plot $500,000,000 homes that only the king of Saudia Arabia can afford? Do they even exist?

Do we really want to keep buildings that have 10,000 units inside? Last I heard the Empire State wasn't for sale.

Maybe if we were more interested in analytics dealing with strictly the most wealthy or crazy corporate skyscrapers. But in this analysis we want to look more closely at modelling for 'average' to 'moderately expensive' sized buildings; those which fall under the highest populated intervals.

### Ranges & Intervals of NYC 

**1)** We want sale prices to be between $100,000 - $5,500,000 because it's simply reasonable.

**2)** We don't want any weird or incorrect values where the year built is 0.

**3)** We don't want any illogical entries creating skewedness in our analysis.

**4)** We want the total units of a building to not have more than 60.

```
#Drop all SALE PRICE outlier values outside a $100,000 to $5,500,000 
data = data[(data[price] > 100000) & (data[price] < 5500000)]

#Drop all 0 YEAR BUILT values
data = data[data[built] > 0]

#Create AGE OF BUILDING to make visualization more meaningful
data[age] = 2017 - data[built]

#Dropping 0 TOTAL UNITS values and deleting outliers in TOTAL UNITS that are above 60
data = data[(data[total] > 0) & (data[total] < 60)] 

#Dropping data where TOTAL NUTS are not a sum of either COMMERCIAL or RESIDENTIAL
data = data[data[total] == data[commercial] + data[residential]]
```

Of course choosing these restraints are completely arbitrary but I chose values that made sense to me.

## III) Data Visualization

We want to make use of a number of various plots to inspect the distributions between features in our data set.

Some of the more interesting plots we want to use are histograms, scatterplots, boxplots, and pairplots. To be able to get an understanding of how the raw data vs the cleaned data looks, I've plotted 'before and after' pictures for educational and entertainment purposes. 

Since the code written for both plots is the same I will refrain from writing each more than once but show the raw version first and cleaned version second.

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_3_Histogram_SalePrice_Uncleaned.png">
    Figure 01: Raw Histogram 
</p>

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_3_Histogram_SalePrice_Cleaned.png">    
    Figure 02: Cleaned Histogram
</p>

```
#Plot SALE PRICE Histogram
sns.set(font_scale=3.1)
hist = sns.distplot(data[price])
plt.title('Histogram of Sale Price')
plt.xlabel('Sale Price (USD)')
plt.ylabel('Normalized Frequency')
plt.show()
```
&nbsp;

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_1_Scatter_(SalePrice-Land)_Uncleaned.png" width="795" height="485">
    Figure 03: Raw Scatterplot Land
</p>

&nbsp;

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_2_Scatter_(SalePrice-Gross)_Uncleaned.png" width="795" height="485">
    Figure 04: Raw Scatterplot Gross
</p>


&nbsp;

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_1_Scatter_(SalePrice-Land)_Cleaned1.png" width="795" height="485">
    Figure 05: Cleaned Scatterplot Land
</p>

&nbsp;

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_2_Scatter_(SalePrice-Gross)_Cleaned1.png" width="795" height="485">
    Figure 06: Cleaned Scatterplot Gross
</p>

```
#Plotting ScatterPlot for each SALE PRICE distribution
sns.set(font_scale=3.1)
g = sns.lmplot(data=data, x=price, y=gross, hue=location, fit_reg=False, size = 10.5, scatter_kws={'s':135}, legend_out = False)
for ax in g.axes.flat:
    plt.setp(ax.get_legend().get_texts(), fontsize='22') # legend text
    plt.setp(ax.get_legend().get_title(), fontsize='27') # legend title
g.despine(left=True)
plt.title('Distribution of Sale Price')
g.set(xlabel='Sale Price (USD)')
g.set(ylabel='Gross Square Feet')
plt.show()
```

&nbsp;

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_4_Boxplot_SalePrice_Uncleaned.png" width="915" height="485">
    Figure 07: Raw Sales Price Boxplot
</p>

&nbsp;

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_4_Boxplot_SalePrice_Cleaned1.png" width="915" height="485">
    Figure 08: Cleaned Sales Price Boxplot
</p>

```
#Plotting BoxPlot for SALE PRICE
sns.set(font_scale=3.1)
box = sns.boxplot(x=price, data=data, linewidth=5, fliersize=12)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of Sale Price')
box.set(xlabel='Sale Price (US)')
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.show()
```
<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_4_PairPlot_Uncleaned.png">
    Figure 09: Raw Pairplot
</p>

&nbsp;

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_4_PairPlot_Cleaned1.png">
    Figure 10: Cleaned Pairplot
</p>

```
#Plotting PairPlot for SALE PRICE on GROSS SQUARE FEET and LAND SQUARE FEET
plt.figure()
sns.set(font_scale=2.3)
pair = sns.pairplot(data, vars=[price, gross, land], size=5, hue=location, aspect=1.2)
plt.show()
```

As you can quite obviously tell the distributions for every plot from Uncleaned to Cleaned are beyond noticable. Not only are we able to visualize the distributions of the sales price but we're also able to better discern what the distributions look like for the housing locations as well.

If we dig further we can also visualize these correlations with respect to their coefficients by building a correlation matrix shown below.


<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/corr_matrix1.png" width="715" height="585">
</p>

```
#Compute the correlation matrix
sns.set(font_scale=2.1)
d = data[[residential, commercial, total, age, price, gross, land]]
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
```

One final bit of information can be attained by looking at the boxplot price distribution in terms of the location of the house. This could provide some insights into where most of the most dollar-value houses are located in NYC.

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_5_Boxplot_Location1.png">
    Figure 12: Boxplot for Sales Price Distribution on Location
</p>

```
#Plot Boxplot Location
sns.boxplot(x=location, y=price, data=data)
plt.title('Sale Price Distribution by Location', fontsize=30)
plt.ylabel('Sale Price (USD)', fontsize=30)
plt.xlabel('Location', fontsize=30)
plt.show()
```
What have we gained from these visualizations?

It would seem as though Manhattan definitely holds the place for highest cost of living because the first quartile begins somewhere in the $1.7 million zone! I guess that's not really surprising.

What's really interesting is the correlation matrix.

We'd need a real-estate specialist on board to help us make sense of the fact that the age of the building is much higher magnitude in correlation to the price as the land or gross square footage; not something I would have expected.


## IV) Tranforming Data for Model Building
### Standardization of Varying Numerical Data

We're almost at the 'cool' part.

Because we are utilizing certain regression models in training our data - Lasso and Ridge - and because our data is of differing varieties and magnitudes - Price (USD), Square Footage, Units, Age - we need to be working with standardized data. This means we need to center our price distribution (dependant variable) along a zero mean and transform all the data to a logarithmic scale like below for our sale price.

This will help our ML tools in Scikit-Learn perform better because if we've scaled our features our models will converge faster - at least that's the hope.

One thing to note is that scaling data for logarithmic data means that we can't have any zeroes in the set because log(0) is undefined. Instead we can treat those zeroes in the original numerical space as zeroes in logarithm space by setting all zero values to 1 since log(1) = 0. So then we have to add 1 to all the columns that have minimum values of 0 or else we're going to get errors.

The minimum column Sale Price value is not zero so we can skip adding a 1 to it.

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

#Standardizing and transforming into logarithmic scale
data[price] = StandardScaler().fit_transform(np.log(data[price]).reshape(-1,1))
```
Now plotting the standardized form of our sale price histogram below.

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_3_Histogram_SalePrice_Cleaned_Standardized.png">
    Figure 13: Standardized Sales Price Histogram
</p>

```
#Plot the new Standardized SALE PRICE
sns.distplot(data[price])
plt.title('Standardized Histogram of Sale Price')
plt.xlabel('Sale Price (USD)')
plt.ylabel('Normalized Frequency')
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
#Splitting categorical data
one_hot_data = [location, build_class]

#Encoding and updating the data
one_hot_encoded = pd.get_dummies(data[one_hot_data])
one_hot_encoded.info(verbose=True, null_counts=True)
data = data.drop(one_hot_data, axis=1)
data = pd.concat([data, one_hot_encoded], axis=1)
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
### Modeling & Choosing the "k" in K-Fold 

Before we begin running our data through we need to choose a k value for a cross-validation. Having a higher k value means that more iterations will take place at a lower sample size. So it's a trade-off.

On one hand if we have a higher k value then we have a higher computational time, higher variance, and lower bias. If we have a smaller k value then we have a lower computational time, lower variance, and higher bias.

We choose k = 5 because we only have ~30,000 data points. However, you should always note how much data you have to work with. Since we deleted a lot of our data we aren't left with a whole lot - so choosing k = 5 seems reasonable.

## Ridge Regression Model

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
```
We have a vector of coefficients which is (100, 41) which is the length of columns of total alpha values by the rows for all of our features.

I want to visualize how these alpha values change the weight of each coefficient as the value of alpha increases.

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_6_Coeff_Ridge.png">
    Figure 14: Coefficient Ridge Weights as function of Alpha
</p>

Because there are 41 features in our modeling set I refrain from labeling them on a legend but we can see that as we choose higher values of alpha the weights of our features in the model decrease - some more rapidly than others.

```
#Plot regularization parameters for Ridge
ax = plt.gca()
sns.set(font_scale=2.3)
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.title(r'Ridge Coefficient Weights with Changing $\alpha$', fontsize=32)
plt.xlabel(r'Regularization Parameters $\alpha$', fontsize=27)
plt.ylabel('Weights', fontsize=27)
plt.xlim((min(alphas), max(alphas))) 
```
Running Scikit-Learn's RidgeCV tool will allow us to find the best alpha value using the mean squared error minimization.
```
#Calculate best alpha with smallest cross-validation error for Ridge
ridgecv = RidgeCV(alphas=alphas, scoring='mean_squared_error', normalize=True)
ridgecv.fit(X_train, y_train)
ridgecv.alpha_
```
This gives us a value of alpha = 0.4348745013088917. Let's put it into or regressor and fit the model.

```
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
<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_7_Coeff_Ridge_Scores.jpg" width="435" height="65">
</p>

Coefficients of determination at about 0.401 is not bad but this is still not the most tuned model we could come up with so it's to be expected. Now let's take a look at the coefficients found from our alpha value.

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_7_Coeff_Ridge_Values.png" width="590" height="590">
</p>

## Lasso Regression Model

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
```
Again we take a look at how the alpha changes the weights on a Lasso model.

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_6_Coeff_Lasso.png">
        Figure 17: Coefficient Lasso Weights as function of Alpha
</p>

This behaviour is 'quickly' convergent compared to the Ridge regressor and only very small values of alpha seem to work here.

```
#Plot regularization parameters for Lasso
ax = plt.gca()
ax.plot(alphas*2, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

```
We want to iterate for Lasso at a maximum value of 10,000 times.

```
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

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_7_Coeff_Lasso_Scores.png" width="415" height="65">
</p>

Coefficient of determination score at about 0.401 is not bad but this is an untuned model so it's to be expected. Now let's take a look at the coefficients found from our alpha value.

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_7_Coeff_Lasso_Values.png" width="590" height="590">
</p>

Surprisingly enough the mean square error is very close to that of the Ridge. I am not entirely sure why that is but it is something for a later investigation.

## Random Forest Model

Last but not least we want to use one decision tree type of model to make some regressive predictions as well.

Random forest will branch subsets of our data into independant, smaller decision trees, and then boost out a weighted average of what features are most important. This is the so-called 'importance' determination of features - which will be described in a more thorough investigation in another text file.

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

<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_8_Coeff_RandomForest_Scores.png" width="415" height="65">
</p>


We now want to showcase the importance that has been measured by our model. We create a new dataframe with this list in descending order to see how valuable each feature actually is.

```
importance = pd.DataFrame(list(zip(X_train.columns, np.transpose(rf_reg.feature_importances_)))).sort_values(1, ascending=False)
print(importance)
```
<p align="center"> 
<img src="https://github.com/gdivanov/NYC-Housing/blob/master/Figures/Figure_8_Coeff_RandomForest_Importances.png" width="420" height="590">
</p>

## VI) Conclusions

For our data set we can only really deduce certain key features of our observations without being an NYC real-estate expert.

In short a list of attributes of our overall study will suffice because this project was meant to showcase a data science step-by-step project without fully committing to heavy details.

So,

**1)** We saw that the Random Forest model gave us the highest coefficient of determination of ~0.433. This means that if we were to begin modeling more that this might be the best choice given that we tuned both the Ridge and Lasso slightly, yet still got outperformed by the Random Forest.

**2** The importances seemed to suggest that the Gross Square Footage of the building far outweighs the rest of the categories. This would need further investigation because if there are some issues with the default importances measures by Scikit-Learn (which is something I investigate in my next study) then further tuning is necessary.

**3** A lot of data was deleted - roughly a whopping **65%** from what we originally had before cleaning which left us only **35%** to work with. This means possible alterations to the data could be made to keep more of it for better results.

