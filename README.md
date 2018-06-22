# NYC-Housing
Visualizing and modeling the correlation between features in New York Department of Finance data set

We want to test a few models in our analysis - namely Random Forest, Linear/Lasso/Ridge regression types.
Note that I include an explanation for Random Forest importance biases in the Scikit-Learn library provided by http://explained.ai/rf-importance/index.html and its packages.

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

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV

#Modeling tool imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
```

```
#Upload data set
data = pd.read_csv('C:\\...\\Python Scripts\\NYC Housing\\nyc-rolling-sales.csv')

#Data Cleaning-----------------------------------------------

#Renamaning columns for convenience
price = 'SALE PRICE'
gross = 'GROSS SQUARE FEET'
land = 'LAND SQUARE FEET'
location = 'BOROUGH'
date = 'SALE DATE'
built = 'YEAR BUILT'

#Matching location data in BOROUGH to numerical values
data[location] = data[location].replace({1: 'Manhattan', 2: 'Bronx', 3: 'Brooklyn', 4: 'Queens', 5: 'Staten Island'})

#Drop entire columns of EASE-MENT and Unnamed: 0 because they are empty
del data['EASE-MENT']
del data['Unnamed: 0']

#Giving all object-oriented data numerical values for analysis
data[price] = pd.to_numeric(data[price], errors='coerce')
data[gross] = pd.to_numeric(data[gross], errors='coerce')
data[land]= pd.to_numeric(data[land], errors='coerce')
data[date] = pd.to_datetime(data[date], errors='coerce')

#Delete the duplicates and null-space data in SALE PRICE
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
age = 'AGE OF BUILDING'
data[age] = 2017 - data[built]

#Dropping 0 TOTAL UNITS values and deleting outliers in TOTAL UNITS that are above 60
data = data[(data['TOTAL UNITS'] > 0) & (data['TOTAL UNITS'] < 60)] 

#Dropping data where TOTAL NUTS are not a sum of either COMMERCIAL or RESIDENTIAL
data = data[data['TOTAL UNITS'] == data['COMMERCIAL UNITS'] + data['RESIDENTIAL UNITS']]

#Visualization ------------------------------------------

#Plot SALE PRICE Histogram for total range
plt.figure()
sns.set(font_scale=3.1)
hist = sns.distplot(data[price])
plt.title('Histogram of Sale Price')
plt.xlabel('Sale Price (USD)')
plt.ylabel('Normalized Frequency')
plt.savefig('fig1.pdf')

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
```
