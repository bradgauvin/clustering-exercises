#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoLars, LinearRegression, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_regression, RFE

import pandas as pd

def plot_histograms(df, columns):
    """ Plots multiple histograms of specified columns argument using data from input df """
    # List of columns
    cols = columns
    plt.figure(figsize=(25, 10))
    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.histplot(data=df[col], bins=10)

        # Hide gridlines.
        plt.grid(False)

    plt.show()

def plot_boxplots(df, columns):
    """ Plots multiple boxplots of specified columns argument using data from input df """
    # List of columns
    cols = columns
    plt.figure(figsize=(16, 6))
    for i, col in enumerate(cols):

        # i starts at 0, but plot nos should start at 1
        plot_number = i + 1 

        # Create subplot.
        plt.subplot(1, len(cols), plot_number)

        # Title with column name.
        plt.title(col)

        # Display boxplot for column.
        sns.boxplot(data=df[col])

        # Hide gridlines.
        plt.grid(False)

    plt.show()

def plot_variable_pairs(df, numerics, categoricals, targets, sample_amt):
    """ Plots pairwise relationships between numeric variables in df along with regression line for each pair. Uses categoricals for hue."""
    # Sampling allows for faster plotting with large datasets at the expense of not seeing all datapoints
    # Checks if a sample amount was inputted
    if sample_amt:
        df = df.sample(sample_amt)
    # Checks if any categorical variables were given to determine how to set the lmplot regression line parameters
    if len(categoricals)==0:
        categoricals = [None]
        # Setting to red makes it easier to see against the default color
        line_kws = {'lw':4, 'color':'red'}
    else:
        line_kws = {'lw':4}
    for cat in categoricals:    
        for col in numerics:
            for y in targets:
                if y == col:
                    continue
                sns.lmplot(data = df, 
                           x=col, 
                           y=y, 
                           hue=cat, 
                           palette='Set1',
                           scatter_kws={"alpha":0.2, 's':10}, 
                           line_kws=line_kws,
                           ci = None)
            

def plot_categorical_and_continuous_vars(df, categorical, continuous, sample_amt):
    """ Accepts dataframe and lists of categorical and continuous variables and outputs plots to visualize the variables"""
    sns.set(font_scale=1.2)
    # Sampling allows for faster plotting with large datasets at the expense of not seeing all datapoints
    if sample_amt:
        df = df.sample(sample_amt)
    # Outputs 3 plots showing high level summary of the inputted data
    for num in continuous:
        for cat in categorical:
            _, ax = plt.subplots(1,3,figsize=(20,8))
            print(f'Generating plots {num} by {cat}')
            # Strip plot
            p = sns.barplot(data = df, x=cat, y=num, ax=ax[0])
            # Mean line
            p.axhline(df[num].mean())
            # Boxplot
            p = sns.boxplot(data = df, x=cat, y = num, ax=ax[1])
            # mean line
            p.axhline(df[num].mean())
            # Violine plot
            p = sns.violinplot(data = df, x=cat, y=num, hue = cat, ax=ax[2])
            #mean line
            p.axhline(df[num].mean())
            plt.suptitle(f'{num} by {cat}', fontsize = 18)
            plt.show()

def plot_pairplot_pairs(df):
    return sns.pairplot(df, corner = True, kind='reg', plot_kws={'line_kws':{'color':'red'}, 'scatter_kws':{'s': 1, 'alpha': 0.5}})

# for hypothesis test
def stats_result(p,null_h,**kwargs):
    """
    Compares p value to alpha and outputs whether or not the null hypothesis
    is rejected or if it failed to be rejected.
    DOES NOT HANDLE 1-TAILED T TESTS
    
    Required inputs:  p, null_h (str)
    Optional inputs: alpha (default = .05), chi2, r, t, corr
    
    """
    #Get alpha value - Default to .05 if not provided
    alpha=kwargs.get('alpha',.05)
    #get any additional statistical values passed (for printing)
    t=kwargs.get('t',None)
    r=kwargs.get('r',None)
    chi2=kwargs.get('chi2',None)
    corr=kwargs.get('corr',None)
    
    #Print null hypothesis
    print(f'\n\033[1mH\u2080:\033[0m {null_h}')
    #Test p value and print result
    if p < alpha: print(f"\033[1mWe reject the null hypothesis\033[0m, p = {p} | α = {alpha}")
    else: print(f"We failed to reject the null hypothesis, p = {p} | α = {alpha}")
    #Print any additional values for reference
    if 't' in kwargs: print(f'  t: {t}')
    if 'r' in kwargs: print(f'  r: {r}')
    if 'chi2' in kwargs: print(f'  chi2: {chi2}')
    if 'corr' in kwargs: print(f'  corr: {corr}')

    return None

def model_feature_selection(train, validate, test, to_dummy, features_to_scale, columns_to_use):
    """ Performs scaling and feature selection using recursive feature elimination. Performs operations on all three inputed data sets. Requires lists of features to encode (dummy), features to scale, and columns (features) to input to the feature elimination. """
    
    # Gets dummy variables
    X_train_exp = pd.get_dummies(train, columns = to_dummy, drop_first=True)
    
    # Gets dummy variables for validate and test sets as well for later use
    X_train = X_train_exp[columns_to_use]
    X_validate = pd.get_dummies(validate, columns = to_dummy, drop_first=True)[columns_to_use]
    X_test = pd.get_dummies(test, columns = to_dummy, drop_first=True)[columns_to_use]
    
    # Scale train, validate, and test sets using the scale_data function from wrangle.pu
    X_train_scaled, X_validate_scaled, X_test_scaled = wrangle.scale_data2(X_train, X_validate, X_test,features_to_scale, scaler_type)
    
    # Set up the dependent variable in a datafrane
    y_train = train[['tax_value']]
    y_validate = validate[['tax_value']]
    y_test = test[['tax_value']]
    
    # Perform Feature Selection using Recursive Feature Elimination
    # Initialize ML algorithm
    lm = LinearRegression()
    # create RFE object - selects top 3 features only
    rfe = RFE(lm, n_features_to_select=3)
    # fit the data using RFE
    rfe.fit(X_train_scaled, y_train)
    # get mask of columns selected
    feature_mask = rfe.support_
    # get list of column names
    rfe_features = X_train_scaled.iloc[:,feature_mask].columns.tolist()
    # view list of columns and their ranking

    # get the ranks
    var_ranks = rfe.ranking_
    # get the variable names
    var_names = X_train_scaled.columns.tolist()
    # combine ranks and names into a df for clean viewing
    rfe_ranks_df = pd.DataFrame({'Var': var_names, 'Rank': var_ranks})
    # sort the df by rank
    rfe_ranks_df.sort_values('Rank')

    return X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test, rfe_features
    
def model(X_train_scaled, X_validate_scaled, X_test_scaled, y_train, y_validate, y_test, rfe_features, show_test = False, print_results = True):
    """ Fits data to different regression algorithms and evaluates on validate (and test if desired for final product). Outputs metrics for each algorithm (r2, rmse) as a Pandas DataFrame. """
    
    ### BASELINE
    
    # 1. Predict tax_value_pred_mean
    tax_value_pred_mean = y_train['tax_value'].mean()
    y_train['tax_value_pred_mean'] = tax_value_pred_mean
    y_validate['tax_value_pred_mean'] = tax_value_pred_mean

    # 2. compute tax_value_pred_median
    tax_value_pred_median = y_train['tax_value'].median()
    y_train['tax_value_pred_median'] = tax_value_pred_median
    y_validate['tax_value_pred_median'] = tax_value_pred_median

    # 3. RMSE of tax_value_pred_mean
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_mean)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_mean)**(1/2)
    if print_results:
        print("RMSE using Mean\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    # 4. RMSE of tax_value_pred_median
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_median)**(1/2)

    if print_results:

        print("RMSE using Median\nTrain/In-Sample: ", round(rmse_train, 2), 
          "\nValidate/Out-of-Sample: ", round(rmse_validate, 2))

    ### OLS Linear Regression
    
    # create the model object
    lm = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm'] = lm.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lm'] = lm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm)**(1/2)

    if print_results:
        print("RMSE for OLS using LinearRegression\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
    
    # predict test
    if show_test:
        
        y_test['tax_value_pred_lm'] = lm.predict(X_test_scaled)
        rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lm)**(1/2)

    # Lasso-Lars
    
    # create the model object
    lars = LassoLars(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lars.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lars'] = lars.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lars)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lars'] = lars.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lars)**(1/2)

    if print_results:

        print("RMSE for OLS using LarsLasso\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)

    # predict test
    if show_test:
        
        y_test['tax_value_pred_lars'] = lars.predict(X_test_scaled)
        rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lars)**(1/2)

    # Tweedie
    
    # create the model object
    glm = TweedieRegressor(power=1, alpha=0)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    glm.fit(X_train_scaled, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_glm'] = glm.predict(X_train_scaled)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_glm)**(1/2)

    # predict validate
    y_validate['tax_value_pred_glm'] = glm.predict(X_validate_scaled)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_glm)**(1/2)
    
    if print_results:
        print("RMSE for GLM using Tweedie, power=1 & alpha=0\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
        
    # predict test
    if show_test:
        
        y_test['tax_value_pred_glm'] = glm.predict(X_test_scaled)
        rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_glm)**(1/2)

    # Polynomial features
    # make the polynomial features to get a new set of features
    pf = PolynomialFeatures(degree=2,interaction_only=True)

    # fit and transform X_train_scaled
    X_train_degree2 = pf.fit_transform(X_train_scaled)

    # transform X_validate_scaled & X_test_scaled
    X_validate_degree2 = pf.transform(X_validate_scaled)
    X_test_degree2 = pf.transform(X_test_scaled)
    
    # create the model object
    lm2 = LinearRegression(normalize=True)

    # fit the model to our training data. We must specify the column in y_train, 
    # since we have converted it to a dataframe from a series! 
    lm2.fit(X_train_degree2, y_train.tax_value)

    # predict train
    y_train['tax_value_pred_lm2'] = lm2.predict(X_train_degree2)

    # evaluate: rmse
    rmse_train = mean_squared_error(y_train.tax_value, y_train.tax_value_pred_lm2)**(1/2)

    # predict validate
    y_validate['tax_value_pred_lm2'] = lm2.predict(X_validate_degree2)

    # evaluate: rmse
    rmse_validate = mean_squared_error(y_validate.tax_value, y_validate.tax_value_pred_lm2)**(1/2)
    
    if print_results:
        print("RMSE for Polynomial Model, degrees=2\nTraining/In-Sample: ", rmse_train, 
          "\nValidation/Out-of-Sample: ", rmse_validate)
        
    # predict test
    if show_test:
        
        y_test['tax_value_pred_lm2'] = lm2.predict(X_test_degree2)
        rmse_test = mean_squared_error(y_test.tax_value, y_test.tax_value_pred_lm2)**(1/2)

    results = pd.concat([
    y_validate.apply(lambda col: r2_score(y_validate.tax_value, col)).rename('r2'),
    y_validate.apply(lambda col: mean_squared_error(y_validate.tax_value, col)).rename('mse'),
    ], axis=1).assign(
        rmse=lambda df: df.mse.apply(lambda x: x**0.5)
    )
    if show_test:
        results = pd.concat([
        y_train.apply(lambda col: r2_score(y_train.tax_value, col)).rename('r2_train'),
        y_train.apply(lambda col: mean_squared_error(y_train.tax_value, col)).rename('mse_train'),
        y_validate.apply(lambda col: r2_score(y_validate.tax_value, col)).rename('r2_validate'),
        y_validate.apply(lambda col: mean_squared_error(y_validate.tax_value, col)).rename('mse_validate'),
        y_test.apply(lambda col: r2_score(y_test.tax_value, col)).rename('r2_test'),
        y_test.apply(lambda col: mean_squared_error(y_test.tax_value, col)).rename('mse_test'),
        ], axis=1).assign(
            rmse_validate=lambda df: df.mse_validate.apply(lambda x: x**0.5)
        )
        
        results = results.assign(rmse_train= lambda results: results.mse_train.apply(lambda x: x**0.5))
        results = results.assign(rmse_test= lambda results: results.mse_test.apply(lambda x: x**0.5))


def select_kbest(X, y, k):
    """ Takes in predictors (X), target (y) , and number of features to select (k) and returns the names 
    of the top k selected features based on the SelectKBest class."""
    # f_regression stats test for top 2
    f_selector = SelectKBest(f_regression, k=k)
    # find the top 2 X's correlated with y
    f_selector.fit(X,y)
    # Boolean mask of whether the column was selected or now
    feature_mask = f_selector.get_support()
    # List of top k features
    return X.iloc[:,feature_mask].columns.tolist()

def rfe(X,y,k):
    """ Takes in predictors (X), target (y) , and number of features to select (k) and returns the names 
    of the top k selected features based on the RFE class."""
    # initialize the ML algorithm
    lm = LinearRegression()

    # create the rfe object, indicating the ML object (lm) and the number of features I want to end up with. 
    rfe = RFE(lm, n_features_to_select=k)

    # fit the data using RFE
    rfe.fit(X,y)  

    # get the mask of the columns selected
    feature_mask = rfe.support_

    # get list of the column names. 
    return X.iloc[:,feature_mask].columns.tolist()

def scale_data2(train, validate, test, features_to_scale, scaler_type):
    """Scales data using MinMax Scaler. 
    Accepts train, validate, and test datasets as inputs as well as a list of the features to scale. 
    Returns dataframe with scaled values added on as columns"""
    
    # Select which scaler to use
    if scaler_type == 'MinMax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    elif scaler_type == 'Standard':
        scaler = sklearn.preprocessing.StandardScaler()
    elif scaler_type == 'Robust':
        scaler = sklearn.preprocessing.RobustScaler()
    else:
        print("Invalid scaler entry, using MinMax")
        scaler = sklearn.preprocessing.MinMaxScaler()
        
    # Fit the scaler to train data only.     
    scaler.fit(train[features_to_scale])
    
    # Generate a list of the new column names with _scaled added on
    scaled_columns = [col+"_scaled" for col in features_to_scale]
    
    # Transform the separate datasets using the scaler learned from train
    scaled_train = scaler.transform(train[features_to_scale])
    scaled_validate = scaler.transform(validate[features_to_scale])
    scaled_test = scaler.transform(test[features_to_scale])
    
    # Concatenate the scaled data to the original unscaled data
    train_scaled = pd.concat([train, pd.DataFrame(scaled_train,index=train.index, columns = scaled_columns)],axis=1).drop(columns = features_to_scale)
    validate_scaled = pd.concat([validate, pd.DataFrame(scaled_validate,index=validate.index, columns = scaled_columns)],axis=1).drop(columns = features_to_scale)
    test_scaled = pd.concat([test, pd.DataFrame(scaled_test,index=test.index, columns = scaled_columns)],axis=1).drop(columns = features_to_scale)
    
    
    return train_scaled, validate_scaled, test_scaled
    

def remove_outliers(df, k, col_list):
    ''' From the class. Removes outliers based on multiple of IQR. Accepts as arguments the dataframe, the k value for number of IQR to use as threshold, and the list of columns. Outputs a dataframe without the outliers.
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df.drop(columns=['outlier'], inplace=True)
    # print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df

def remove_outliers_std(df, k, col_list):
    ''' Removes outliers based on multiple of standard devisation from mean. Accepts as arguments the dataframe, the k value for number of standard deviations to use as threshold, and the list of columns. Outputs a dataframe without the outliers.
    '''
    # Create a column that will label our rows as containing an outlier value or not
    num_obs = df.shape[0]
    df['outlier'] = False
    for col in col_list:

        average = df[col].mean()  # get mean
        standard_deviation = df[col].std() # get standard deviation
        
        upper_bound = average + k * standard_deviation   # get upper bound
        lower_bound = average - k * standard_deviation  # get lower bound

        # update the outlier label any time that the value is outside of boundaries
        df['outlier'] = np.where(((df[col] < lower_bound) | (df[col] > upper_bound)) & (df.outlier == False), True, df.outlier)
    
    df = df[df.outlier == False]
    df.drop(columns=['outlier'], inplace=True)
    # print(f"Number of observations removed: {num_obs - df.shape[0]}")
        
    return df
## TODO How to handle 0 bedroom, 0 bathroom homes? Drop them? How many? They're probably clerical nulls

