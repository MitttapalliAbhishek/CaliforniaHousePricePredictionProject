#!/usr/bin/env python
# coding: utf-8

# # Python California Housing Project

# In[1]:


# importing necessary libraries/modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading a csv file using pandas packages
df = pd.read_csv("housing.csv")
df.head()


# In[3]:


print(type(df))


# In[4]:


df.tail()


# In[5]:


df[["housing_median_age","total_rooms"]].head()


# In[6]:


df.total_rooms


# ### Information About the Housing Dataset

# In[7]:


df.info()


# In[8]:


df[df.housing_median_age >= 20]


# In[9]:


df.describe()


# In[10]:


df["ocean_proximity"].value_counts()


# In[11]:


df.shape


# In[12]:


df.columns


# ###  Plotting

# In[13]:


df.hist(bins = 50,figsize = (15,8))
plt.show()


# In[14]:


df.head()


# In[15]:


df.plot(kind = "scatter",x = "longitude",y = "latitude",cmap = "jet")
plt.show()


# In[16]:


df["median_income"].hist()


# In[17]:


# dividong the income catogeroy to limit the number income category
df["income_cat"] = np.ceil(df["median_income"] / 1.5)

#putting everything above fifth category as fifth category
df["income_cat"].where(df["income_cat"] < 5,other = 5.0,inplace = True)


# In[18]:


df.head()


# In[19]:


# Using stratified sampling
from sklearn.model_selection import StratifiedShuffleSplit
# Splitting into train and test sets

data_split = StratifiedShuffleSplit(n_splits = 1,test_size = 0.2,random_state = 30)
for train_index, test_index in data_split.split(df,df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]


# In[20]:


df["income_cat"].value_counts()


# In[21]:


# Propotions of categories split in income_cat feature
df["income_cat"].value_counts() / len(df)


# In[22]:


strat_train_set["income_cat"].value_counts() / len(strat_train_set)


# In[23]:


# using random sampling
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df,test_size = 0.2,random_state = 30)


# In[24]:


# comparing stratified sampling with random sampling
def income_cat_propotions(data):
    return data["income_cat"].value_counts() / len(data)

compare = pd.DataFrame({
    "Overall propotions":income_cat_propotions(df),
    "Random Sampling": income_cat_propotions(test_set),
    "Stratified Sampling": income_cat_propotions(strat_test_set)
}).sort_index()


# In[25]:


compare["Random Sampling error in %"] = 100 * compare["Random Sampling"] / compare["Overall propotions"] - 100
compare["Stratified Sampling error in %"] = 100 * compare["Stratified Sampling"] / compare["Overall propotions"] - 100
compare


# In[26]:


for values in (strat_train_set,strat_test_set):
    values.drop("income_cat",axis = 1,inplace = True)


# In[27]:


housing = strat_train_set.copy()


# In[28]:


housing.head()


# In[29]:


housing.plot(kind = "scatter",x = "longitude",y = "latitude",alpha = 0.4,
            s = housing['population']/100,label = "population",figsize = (12,8),
            c = "median_house_value",cmap = plt.get_cmap("jet"),sharex=False)


plt.legend()


# In[30]:


housing.plot(kind = "scatter",x = "longitude",y = "latitude",alpha = 0.4,
            s = housing['population']/100,label = "population",figsize = (12,8),
            c = "median_house_value",cmap = plt.get_cmap("jet"),sharex=False)

plt.title("California Map")
plt.legend()

# Adding the plot on top of the google image
import matplotlib.image as mpimg

california_img = mpimg.imread("california.png")
plt.imshow(california_img, extent=[-125.65, -114.8, 32.45, 42.05], 
           alpha=0.5, cmap=plt.get_cmap("jet"))
plt.xlabel("Longitude",fontsize = 14)
plt.ylabel("Latitude",fontsize = 14)
plt.legend(fontsize = 14)
plt.show()


# ### Finding correlations of the features

# In[31]:


# finding correlations of the california housing features
# correlations range from -1 to 1 where 1 means strong corrleation and -1 is weak correlation
corr_matrix = housing.corr()
corr_matrix


# In[32]:


corr_matrix["median_house_value"].sort_values(ascending = False)


# ### Correlation Matrix using Heatmap

# In[33]:


import seaborn as sns

corr_graph_matrix = sns.heatmap(corr_matrix,annot = True)
corr_graph_matrix


# ### Correlation Matrix using Scatter Matrix

# In[34]:


from pandas.plotting import scatter_matrix
imp_attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[imp_attributes],figsize = (15,8))
plt.show()


# In[35]:


housing.plot(kind = "scatter",x = "median_income",y = "median_house_value",alpha = 0.1)
plt.axis([0,16,0,550000])


# ### Feature Engineering

# In[36]:


# creating a new features in our housing dataframe
housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]


# In[37]:


housing.head()


# In[38]:


# Againg finding correlations of the features 
corr_matrix = housing.corr()
corr_matrix


# In[39]:


corr_matrix["median_house_value"].sort_values(ascending = False)


# ### Preparing data for Machine Learning Algorithms

# In[40]:


housing = strat_train_set.drop("median_house_value",axis = 1)
housing_labels = strat_train_set["median_house_value"].copy()


# In[41]:


housing.head()


# In[42]:


housing_labels.head()


# In[43]:


print("Therefore there are "+str(housing["total_bedrooms"].isna().sum())+" missing values in total_bedroom column in housing dataset")


# In[44]:


# Using SimpleImputer to fit all the nan values present in the dataframe
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")


# In[45]:


housing_num = housing.drop("ocean_proximity",axis = 1)
imputer.fit(housing_num)


# In[46]:


imputer.statistics_


# In[47]:


X = imputer.transform(housing_num)


# In[48]:


housing_tr = pd.DataFrame(X,columns = housing_num.columns)
housing_tr.head()


# In[49]:


# cross checking if any missing values are present or not in the dataset
housing_tr["total_bedrooms"].isna().sum()


# ### Handling Categorical Values present in the dataset

# In[50]:


housing_cat = housing["ocean_proximity"]
housing_cat.head(10)


# In[51]:


# using pandas's own factorize() method to convert them into categorical features
housing_cat_encoded, housing_categories = housing_cat.factorize()


# In[52]:


# label encoded categories
housing_cat_encoded[:10] 


# In[53]:


housing_categories


# In[54]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(1,-1))


# In[55]:


# returning sparse matrix
housing_cat_1hot


# In[56]:


# since 1 hot encoder returns a sparse matrix, need to change it to a dense array
housing_cat_1hot.toarray()


# ### Custom Transformations

# In[57]:


from sklearn.base import BaseEstimator, TransformerMixin
rooms_index, bedrooms_index, population_index, households_index = 3,4,5,6
# Transformer class for numerical attributes

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
    def __init__(self,add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self,X,y = None):
        return self
    
    def transform(self,X,y = None):
        rooms_per_household = X[:,rooms_index] / X[:,households_index]
        population_per_household = X[:,population_index] / X[:,households_index]
        if self.add_bedrooms_per_room:
            bedrooms_per_household = X[:,bedrooms_index] / X[:,rooms_index]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_household]
        else:
            return np.c_[X,rooms_per_household,population_per_household]


# In[58]:


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room = False)
housing_extra_attribs = attr_adder.transform(housing.values)


# In[59]:


housing_extra_attribs = pd.DataFrame(housing_extra_attribs,columns = list(housing.columns)+["rooms_per_household","population_per_household"])
housing_extra_attribs.head()


# ### Setting up the pipline for numerical processesing

# In[60]:


# This pipeline if for processing numerical attributes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ("imputer",SimpleImputer(strategy = "median")),
    ("attr_adder",CombinedAttributesAdder()),
    ("std_scalar",StandardScaler())
])


# In[61]:


housing_num_tr = num_pipeline.fit_transform(housing_num)
housing_num_tr


# In[62]:


# Transformer class for categorical attributes
class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names = attribute_names
    
    def fit(self,X,y = None):
        return self
    
    def transform(self,X,y = None):
        return X[self.attribute_names].values


# In[63]:


# setting up complete pipeline
num_attribs = list(housing_num.columns)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ("selector",DataFrameSelector(num_attribs)),
    ("imputer",SimpleImputer(strategy = "median")),
    ("attr_adder",CombinedAttributesAdder()),
    ("std_scalar",StandardScaler())
])

cat_pipeline = Pipeline([
    ("selector",DataFrameSelector(cat_attribs)),
    ("cat_encoder",OneHotEncoder(sparse = False))
])


# In[64]:


from sklearn.pipeline import FeatureUnion
full_pipeline = FeatureUnion(transformer_list = [
    ("num_pipeline",num_pipeline),
    ("cat_pipeline",cat_pipeline)
])


# In[65]:


housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared


# In[66]:


housing_prepared.shape


# In[67]:


housing_num_tr.shape


# In[68]:


final_data = pd.DataFrame(housing_prepared,columns = list(housing_num.columns)+["rooms_per_household","population_per_household","bedrooms_per_room"]+list(housing_categories))
final_data.head()


# In[69]:


print("The total number of features present are: ")
for i,j in enumerate(final_data):
    print(i,j)


# In[70]:


housing_labels.head()


# ### Selecting, Building and Training Machine Learning Models

# In[71]:


# This is Linear Regression model
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(housing_prepared,housing_labels)


# In[72]:


# trying the full pipeline on a few training instances
some_data = housing.iloc[:5]
some_data_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)


# In[73]:


print("Predicted values: "+str(model1.predict(some_data_prepared)))
print("Actual values: "+str(list(some_data_labels)))


# In[74]:


score1 = model1.score(some_data_prepared,some_data_labels) * 100
print("The score of the model1 is: "+str(score1)+"%")


# In[75]:


from sklearn.metrics import mean_squared_error

housing_predictions = model1.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("The root mean squared error for LinearRegression model is: "+str(lin_rmse))


# In[76]:


# This is DecisionTreeRegressor model
from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor()
model2.fit(housing_prepared,housing_labels)


# In[77]:


score2 = model2.score(some_data_prepared,some_data_labels) * 100
print("The socre of the model2 is: "+str(score2)+"%")


# In[78]:


housing_predictions = model2.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("The root mean squared error for DecisionTreeRegressor model is: "+str(tree_rmse))


# #### Note:-  therefore the DecisionTreeRegressor model overfitted the data which was fit by the model as tt has root mean squared error as 0.0 and score as 100%

# ### Cross Validation

# In[79]:


from sklearn.model_selection import cross_val_score
# computing socres for LinearRegression model
linear_scores = cross_val_score(model1,housing_prepared,housing_labels,cv = 10,scoring = "neg_mean_squared_error")
linear_rmse_scores = np.sqrt(-linear_scores)


# In[80]:


# computing scores for DecisionTreeRegressor model
tree_scores = cross_val_score(model2,housing_prepared,housing_labels,cv = 10,scoring = "neg_mean_squared_error")
tree_rmse_scores = np.sqrt(-tree_scores)


# In[81]:


def display_scores(scores):
    print("Scores:"+str(scores))
    print("Mean score: "+str(scores.mean()))
    print("Standard deviation of score: "+str(scores.std()))


# In[82]:


# Scores for model1:-
display_scores(linear_rmse_scores)


# In[83]:


# Scores for model2:-
display_scores(tree_rmse_scores)


# In[84]:


# This is RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
model3 = RandomForestRegressor()
model3.fit(housing_prepared,housing_labels)


# In[85]:


print("The Predicted values are: "+str(model3.predict(some_data_prepared)))
print("The Actual values ar: "+str(list(some_data_labels)))


# In[86]:


score3 = model3.score(some_data_prepared,some_data_labels) * 100
print("The score of model3 is: "+str(score3)+"%")


# In[87]:


housing_predictions = model3.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels,housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print("The root mean squared error for RandomForestRegressor is: "+str(forest_rmse))


# In[88]:


# Scores for model3:-
forest_scores = cross_val_score(model3,housing_prepared,housing_labels,cv = 10,scoring = "neg_mean_squared_error")
forest_rmse_scores = np.sqrt(-forest_scores)


# In[89]:


display_scores(forest_rmse_scores)


# ### Fine Tuning Model

# In[90]:


# Hyperparameter tuning using RandomizedSearchCv
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_d = {
    "n_estimators": randint(low = 1,high = 200),
    "max_features": randint(low = 1,high = 8)
}

final_model = RandomForestRegressor(random_state = 30)
r_search = RandomizedSearchCV(final_model,param_distributions = param_d,
                              n_iter = 10,cv = 5,scoring = "neg_mean_squared_error",
                             random_state = 30)

r_search.fit(housing_prepared,housing_labels)


# In[91]:


cv_results = r_search.cv_results_
for mean_score,params in zip(cv_results["mean_test_score"],cv_results["params"]):
    print(np.sqrt(-mean_score),params)


# In[92]:


feature_importances = r_search.best_estimator_.feature_importances_
feature_importances


# In[93]:


extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = cat_pipeline.named_steps["cat_encoder"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


# In[94]:


best_model = r_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = best_model.predict(X_test_prepared)


# In[95]:


score4 = best_model.score(X_test_prepared,y_test) *100
print("The score for best_model is: "+str(score4)+"%")


# In[96]:


final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("The root mean squared error for best_model is: "+str(final_rmse))


# ### Saving the selected  model

# In[97]:


from joblib import dump,load
# dumping the model in .joblib file extension 
dump(model3,"RandomForestRegressor.joblib")


# ### Using the saved model

# In[98]:


# reusing the saved RandomForestRegressor model for predictions
RFR_model = load("RandomForestRegressor.joblib")
input_features = np.array([[7.59846054e-02, 6.91946182e-02, 4.12638793e-02, 1.82267007e-02,
       1.63954706e-02, 1.79034682e-02, 1.60368632e-02, 3.19715471e-01,
       5.98200181e-02, 1.03586792e-01, 8.09190838e-02, 1.25893174e-02,
       1.59214482e-01, 9.70051820e-05, 3.06596698e-03, 5.98625797e-03]])


# In[99]:


# predicting the dependant variable based on the inputted independant variables
predicted_val = RFR_model.predict(input_features)
print("The Predicted house value is: "+str(predicted_val[0])+"$")


# ####  Note:-  The predicted value is in US $

# In[100]:


print("This is project based on predicting prices of houses in california")


# ### The End

# In[ ]:




