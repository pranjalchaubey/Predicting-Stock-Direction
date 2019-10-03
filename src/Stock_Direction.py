# Debug Flag
_GLOBAL_DEBUG_ = False

# yFinance will help us fetch the data for our dataset
import yfinance as yf
# Data Mining and plotting 
import numpy as np
import pandas as pd
# Classifier 
from sklearn.ensemble import RandomForestClassifier


def fetch_ticker_data(tickers): 
    """
    Fetch 5 years of historical data for the 'tickers', from Yahoo Finance.    

    Parameters
    ----------
    tickers : List   
        List of stocks chosen by the user.   
        
    Returns
    -------
    data : Dataframe 
        Pandas df with historical stock data time series. 
    """
    # Debug 
    _LOCAL_DEBUG_ = False

    # We are going to get 10 years worth of stock data
    # Generate the required timestamps 
    t_now = pd.datetime.now().date()
    t_10_year = (t_now - pd.DateOffset(n=3650)).date()
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print(t_now)
        print(t_10_year)

    # Get the Data from Yahoo! Finance
    data = yf.download(tickers, start=t_10_year, end=t_now)
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print(data.head(20))

    # Data is multi-indexed on the columns 
    # Make it multi-index on the rows, to make it 
    # fit for consumption by the RandomForestClassiffier
    data = data.stack(1) 
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print(data.head(20))

    # Drop all the columns except for Adj Close & Volume
    data = data[['Adj Close', 'Volume']]
    # Rename the column names 
    data.columns = ['close', 'volume']  
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print(data.head(20))

    return data

# ## Data Preprocessing
# Now that we have the data downloaded and neatly organized in a dataframe,  
# time to do some necessary preprocessing.  
# In particular, we are going to use the _log returns_ of the stocks to create a  
# target column, called `target`.  
# The target variable will be created using the following criteria  
# ```
#   -1 = Sell = ret < -0.0015
#    0 = Hold = -0.0015 < ret < 0.0015 
#    1 = Buy  = ret > .0015
# ```
def classify_return(ret):
    """
    Classify the returns as -1, 0 or 1.
    -1 = Sell = ret < -0.0015
    0 = Hold = -0.0015 < ret < 0.0015 
    1 = Buy  = ret > .0015

    Parameters
    ----------
    ret : Float   
        Stock return for the current day, d. 
        retd = log(retd) - log(ret(d-1))

    Returns
    -------
    ret_category : Integer
        Category of return, ret. 
        -1, 0 or 1. 
    """
    ret_category = 0
    
    if ret < -0.0015: 
        ret_category = -1
    elif -0.0015 < ret and ret < 0.0015:
        ret_category = 0
    elif ret > 0.0015:
        ret_category = 1

    return ret_category

def data_preprocess(dataset):
    """
    Calculates log returns and creates the 'target' column.
    Returns the processed dataset as well as the data for the current 
    date separately on which the prediction has to be performed. 

    Parameters
    ----------
    dataset : Pandas Dataframe
        Dataframe containing price, volume and other 
        information of the chosen stocks. 

    Returns
    -------
    dataset: DataFrame
        Processed dataframe with two additional columns,
        'returns' and 'target'. 
        returns = log returns of prices 
        target = target column for the classifier to predict
    pred_data: DataFrame
        Data for the 'current date' on which the trained classifier 
        will perform the prediction. The classifier output from 
        this prediction will be visible to the user. 
    """
    # Debug Flag
    _LOCAL_DEBUG_ = False

    # Advance the prices by 1 day to calculate the log returns 
    dataset['shift'] = dataset.groupby(level=1)['close'].shift(1)
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print(dataset.head(30))

    # Create the 'returns' column
    dataset['returns'] = np.log(dataset['close']) - np.log(dataset['shift'])
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        #print('Return Head')
        #print(dataset.head(30))
        print('Return Tail')
        print(dataset.tail(30))

    # Drop the first row as it contains NANs in the returns column
    dataset.drop(index = dataset.index.levels[0].values[0], level=0, inplace=True)
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print(dataset.head(30))

    # Create the 'target' column for the classifier 
    # Shift the 'target' into the future/delay by 1 day 'shift(-1)'
    # We do this since we are predicting returns of the 'next day close'
    dataset['target'] = dataset['returns'].apply(lambda x: classify_return(x))
    dataset['target'] = dataset.groupby(level=1)['target'].shift(-1)
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        #print('Target Head')
        #print(dataset.head(30))
        print('Target Tail')
        print(dataset.tail(30))

    # Drop the 'shift' column as it was only temporary to calculate 'returns'
    dataset.drop(['shift'], axis=1, inplace=True)
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print(dataset.head(30))

    # Extract the data for the current date, before dropping NANs 
    pred_data = dataset.iloc[-(len(ticker_list)):]
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print('pred_data')
        print(pred_data)

    # Handle NANs
    # We keep it simple at the moment and simply drop off rows with NANs 
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print(dataset.shape)
        print(dataset.isna().sum())
    dataset.dropna(how='any', inplace=True)
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print(dataset.shape)
        print(dataset.isna().sum())
        print('Final Dataset Tail')
        print(dataset.tail(30))

    return dataset, pred_data

def train_test_split(dataset, features, target, train_size, test_size):
    """
    Generate the train and test dataset.

    Parameters
    ----------
    dataset : DataFrame
        All the samples including target
    features : List
        List of the names of columns that are features
    target : String
        Name of column that is the target (in our case, 'target')
    train_size : float
        The proportion of the data used for the training dataset
    test_size : float
        The proportion of the data used for the test dataset

    Returns
    -------
    x_train : DataFrame
        The train input samples
    x_test : DataFrame
        The test input samples
    y_train : Pandas Series
        The train target values
    y_test : Pandas Series
        The test target values
    """
    # Data Sanity check 
    assert train_size >= 0 and train_size <= 1.0, 'Train size out of bounds!'
    assert test_size >= 0 and test_size <= 1.0, 'Test size out of bounds!'
    assert train_size + test_size == 1.0, 'Train + Test should be equal to 1!'
    
    # Debug Flag 
    _LOCAL_DEBUG_ = True 
    
    # Extract the x and y from the dataset
    all_x = dataset[features]
    all_y = dataset[target]
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print(all_x.head())
        print(all_y.head())
        print('dataset.shape ', dataset.shape)

    # Get the number of rows in df and no of elements in the pandas series 
    # NOTE - Both are multi-indexed
    len_x = len(all_x.index.levels[0])
    len_y = len(all_y.index.levels[0])
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print('Len x ', len_x)
        print('Len y ', len_y)
    
    # Some fancy calculations here for x
    x_train = all_x.loc[all_x.index.levels[0][:int(len_x*train_size)].astype(str).tolist()]
    x_test = all_x.loc[all_x.index.levels[0][int(len_x*train_size):int(len_x*train_size) + int(len_x*test_size)].astype(str).tolist()]
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print('x_train.shape ', x_train.shape)
        print('x_test.shape ', x_test.shape)

    # Some fancy calculations here for y as well
    y_train = all_y.loc[all_y.index.levels[0][:int(len_y*train_size)].astype(str).tolist()]
    y_test = all_y.loc[all_y.index.levels[0][int(len_y*train_size):int(len_y*train_size) + int(len_y*test_size)].astype(str).tolist()]
    if _GLOBAL_DEBUG_ and _LOCAL_DEBUG_:
        print('y_train.shape ', y_train.shape)
        print('y_test.shape ', y_test.shape)
 
    return x_train, x_test, y_train, y_test 

# ## Make the Prediction
# Now that we have trained our model with the best set of hyperparameters, time  
# to make the prediction for the next date (that is tomorrow's close).  
# The output from this prediction will be shown to the user. 
def make_prediction(pred_data, features, rfc_classifier):
    """
    Makes a price prediction on the stocks as selected by the user. 
    
    Parameters
    ----------
    pred_data : Dataframe
        This dataframe contains only 1 date, today's date (or yesterday's if 
        the markets haven't close yet). The data on this date will be used to 
        make a prediction about the closing prices for next day/tomorrow.
    feature : List
        List of features present in pred_data. It also has a 'target' column, 
        we wont be using it. 
    rfc_classifier : Random Forest Classifier Object 
        Trained Random forest classifier with the best hyperparameters  

    Returns
    -------
    pred_list: List 
        List containing tuples in the following format, 
        [(Stock1, Pred1), (Stock2, Pred2),........]
    """
    # Make the prediction 
    prediction = rfc_classifier.predict(pred_data[features])
    # Create the 'Prediction List' 
    pred_list = [(stock, pred) for stock, pred in zip(pred_data.index.levels[1].values.tolist(), prediction)]

    return pred_list

def main_function(ticker_list):
    """
    This is the main function that makes the predictions. 
    
    Parameters
    ----------
    ticker_list : List
        List of stocks chosen by the user.

    Returns
    -------
    final_prediction: List 
        List containing tuples in the following format, 
        [(Stock1, Pred1), (Stock2, Pred2),........]
    """
    # Fetch the stock prices and build the dataset
    dataset = fetch_ticker_data(tickers=ticker_list)
    # Preprocess the data
    # pred_data the date for which we will be making the 
    # prediction for
    dataset, pred_data = data_preprocess(dataset)
    # In our dataset all the columns except for the 'target' column are features
    # Temporarily drop the 'target' and extract the features  
    features = dataset.drop(['target'], axis=1).columns.values.tolist()
    # Get the train and test sets 
    X_train, X_test, y_train, y_test = train_test_split(dataset=dataset,
                                                        features=features, 
                                                        target='target',
                                                        train_size=1.0,
                                                        test_size=0.0)
    # Train the Random Forest Classifier on this data 
    # Hyperparameters 
    clf_random_state = 0    # This is to get consistent results between each run
    n_days = 5
    n_stocks = len(ticker_list)
    n_trees = 25    # We have selected n_trees to be 25
    clf_parameters = {
        'criterion': 'entropy',
        'min_samples_leaf': n_stocks * n_days,
        'oob_score': True,
        'n_jobs': -1,
        'random_state': clf_random_state}
    # Create the RFC model    
    clf = RandomForestClassifier(n_trees, **clf_parameters)
    # Train 
    clf.fit(X_train, y_train)
    # Predict 
    final_prediction = make_prediction(pred_data=pred_data, features=features,
                                       rfc_classifier=clf)

    return final_prediction


# List of stock tickers 
# this info will come from the user, perhaps in the form of a pickle file 
ticker_list = ['FB', 'AAPL', 'AMZN', 'NFLX', 'GOOGL', 'SBUX', 'XOM', 'JNJ', 'BAC', 'GM']
# Call the Main Function to get the predictions     
f_pred = main_function(ticker_list=ticker_list)
# Print the output 
# In the app, this is the output the user will see 
print(f_pred)    