California-House-Price-Prediction
This is a regression problem to predict california housing prices.

The dataset contains 20640 entries and 10 variables.

Longitude
Latitude
Housing Median Age
Total Rooms
Total Bedrooms
Population
Households
Median Income
Median House Value
Ocean Proximity
Median House Value 


NOTE:- The Median House Value is the target variable in this dataset or it's value need to be predicted.

And We have added additional columns such as rooms_per_household,bedrooms_per_room,population_per_household for better EDA purposes.

We have done this project in two parts. First part contains data analysis and cleaning as explained in EDA in California_Housing_Project.ipynb 
file and new_dataset.ipynb consists of transoformed dataset which is used in training and testing of the machine learning model.

1) EDA and Data Cleaning
We have done the exploratory data analysis and done following manipulations on data.

Creating new features
Removing outliers
Transforming skewed features
Checking for multicoliniearity

2) Training machine learning algorithms:
Here, We have trained various machine learning algorithms like

i)LinearRegression
ii)DecisionTreeRegressor
And using Bagging technique like:
iii)RandomForestRegressor
