from __future__ import division
import csv, sys, re, timeit, math
from sklearn import datasets, linear_model, preprocessing, neural_network
from sklearn.utils import column_or_1d
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import os
import errno
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from matplotlib import dates as mPlotDATEs
# helper functions num2date()
#                                            #              and date2num()
#                                            #              to convert to/from.
# http://stackoverflow.com/questions/4090383/plotting-unix-timestamps-in-matplotlib
# http://stackoverflow.com/questions/32728212/how-to-plot-timestamps-in-python-using-matplotlib 
# http://stackoverflow.com/questions/8409095/matplotlib-set-markers-for-individual-points-on-a-line
rt_start = timeit.default_timer()

# clean log.txt first

directory = "logs"
try:
    os.makedirs(directory)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

log_timestr = datetime.now().strftime("%Y-%m-%d_%H%M%S")
with open("logs/log_" + log_timestr +".txt", "w") as logfile:
    logfile.close()
    
### functions definition ###
def print_data_type(x):
    for f in x.columns:
        print("f = {}".format(f))
        print(x[f].dtype)
        
def RepresentsInt(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False

def RepresentsFloat(s):
    try: 
        float(s)
        return True
    except ValueError:
        return False

def DateToMinute(s, option):
    """
    time delta from the beginning of that year, unit = minutes
    s format = '%Y-%m-%d %H:%M'
    return type: int
    """
    if (isinstance(s, str) ):
        time_obj = datetime.strptime(s, '%Y-%m-%d %H:%M')
    elif (isinstance(s, datetime) ):
        time_obj = datetime.strptime( s.strftime('%Y-%m-%d %H:%M'), '%Y-%m-%d %H:%M' )
    else:
        raise SystemError("input s is not a valid string/datetime obj!")
        
    if (option == 'year'):
        time_diff = time_obj - datetime(time_obj.year, 01, 01, 0, 0)
    elif (option == 'month'):
        time_diff = time_obj - datetime(time_obj.year, time_obj.month, 01, 0, 0)
    elif (option == 'day'):
        time_diff = time_obj - datetime(time_obj.year, time_obj.month, time_obj.day, 0, 0)
    elif (option == 'hour'):
        time_diff = time_obj - datetime(time_obj.year, time_obj.month, time_obj.day, time_obj.hour, 0)
    else:
        raise SystemError("option is not a valid string!")
        
    return int(time_diff.total_seconds()/60)

# do interpolate
def interpolate_df(df, features):
    df_re = df
    
    print("len(df.index) = {}".format(len(df.index)))
    
    # check all the data are float data and change data type to float64
    for col in features:
        # df[col] = df[col].astype(float)
        temp = df[df[col].isnull()]
        # print(test.head)
        print("===")
        # print(test.head(n=1))
        print("{} type is {}".format(col, df[col].dtype))
        print("{} type contain {} np.NaN".format(col, len(temp.index)))
        print("===")
    
    df_nan = df[ df.isnull().any(axis=1) ]
    print("len(df_nan.index) = {}".format(len(df_nan.index)))
    # df_nan.to_csv("df_nan.csv")
    df_nan.head(n=1)
    
    print("len(df.index) = {}".format(len(df.index)))
    # it could be use time as index and set method = 'time'
    # df.to_csv("df_before_interpolate.csv")
    # df[features] = df[features].interpolate(method='time')
    # df.loc[:, features] = df[features].interpolate(method='time')
    # somehow, df(input) will get updated even use inplace=False
    df_re.loc[:, features] = df[features].interpolate(method='time', inplace = False)
    # df.to_csv("df_after_interpolate.csv")
    # print("df = ")
    # print(df)
    
    # grab original nan values
    df_nan_interpolate = df.loc[ df_nan.index.values ]
    print("len(df_nan_interpolate.index) = {}".format(len(df_nan_interpolate.index)))
    df_nan_interpolate.to_csv("df_nan_interpolate.csv")
    
    if (df_re.notnull().all(axis=1).all(axis=0)):
        print("CHECK: There is no null value in df_re.")
        
    return df_re

# generate training & test data set
def data_gen(df, targets, features, data_tr_yr_start, data_tr_yr_end, data_test_yr_start, data_test_yr_end):
    # reset index 
    # df = df.reset_index(drop=True)
    df = df.set_index("DATE")
    # prepare training data
    data_start = datetime(data_tr_yr_start,  1,  1,  0,  0,  0)
    data_end   = datetime(data_tr_yr_end, 12, 31, 23, 59, 59)
    df_train = df.loc[(df.index > data_start ) & (df.index <= data_end ), :]
    
    # do interpolate on training set only
    df_train = interpolate_df(df_train, features)
    df_train.to_csv('df_train_clean.csv')
    
    X_train = df_train[features] 
    y_train = df_train[targets] 
    
    # prepare test data
    data_start = datetime(data_test_yr_start,  1,  1,  0,  0,  0)
    data_end   = datetime(data_test_yr_end, 12, 31, 23, 59, 59)
    df_test = df.loc[(df.index > data_start ) & (df.index <= data_end ), :]
    
    # drop NaN number rows of test set
    (row_old, col_old) = df_test.shape
    print("Before drop NaN number of test set, df_test.shape = {}".format(df_test.shape))
    df_test = df_test[ df_test.notnull().all(axis=1) ]
    (row, col) = df_test.shape
    print("After drop NaN number of test set, df_test.shape = {}".format(df_test.shape))
    print("Drop rate = {0:.2f} ".format(float(1 - (row/row_old)) ) )
    
    df_test.to_csv('df_test_clean.csv')
    X_test = df_test[features] 
    y_test = df_test[targets] 
    
    # normalization and scale for training/test set
    # use robust_scaler to avoid misleading outliers
    # scaler = preprocessing.StandardScaler()
    # use robust_scaler to avoid misleading outliers
    scaler = preprocessing.RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (X_train, y_train, X_test, y_test)
    
def normalization(df_train, df_test, targets, features):
    
    # do interpolate on training set only
    df_train_local = interpolate_df(df_train, features)
    
    X_train = df_train_local[features]
    y_train = df_train_local[targets] 
    
    X_test = df_test[features] 
    y_test = df_test[targets]
    
    # normalization and scale for training/test set
    # use robust_scaler to avoid misleading outliers
    # scaler = preprocessing.StandardScaler()
    # use robust_scaler to avoid misleading outliers
    scaler = preprocessing.RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return (X_train, y_train, X_test, y_test)

# plot y_test
def plot_y_test(regr, X_test, y_test, ask_user):
    (r_test, c_test) = X_test.shape
    
    # for i in range(c_test):
    #     plt.scatter(X_test[:, i], y_test)
    #     plt.plot(X_test[:, i], regr.predict(X_test), color='blue', linewidth=3)
    
    y_predict = regr.predict(X_test)
    # print("==> y_test type = {}".format(type(y_test)) )
    # print("y_test.index = {}".format(y_test.index))
    # print("y_test = {}".format(y_test) )
    # print("y_predict = {}".format(y_predict) )
    df_plot = y_test
    # print(df_plot)
    # print("DATE")
    # print("#########################")
    df_plot = df_plot.reset_index(level=['DATE'])
    df_plot.loc[:,'predict_temp_C'] = y_predict
    # shift back to raw DATE time 1day_later
    df_plot.loc[:,"raw_DATE"] = df_plot['DATE'].apply(lambda time_obj: time_obj + relativedelta(days=1))
    df_plot.rename(columns={'1days_later_temp_C': 'raw_temp_C', 'DATE':'label_DATE'}, inplace=True)
    
    df_plot = df_plot.set_index("raw_DATE")
    # print(df_plot)
    # print("#########################")
    
    # default plot time range
    plot_yr = 2016
    plot_month = 10
    plot_day = 5 
    duration = 10 
    
    range_start = datetime(plot_yr, plot_month, plot_day, 0, 0, 0)
    range_end = datetime(plot_yr, plot_month, plot_day, 0, 0, 0) + relativedelta(days=duration)
    
    if (range_start < datetime(2016,1,2,0,0,0) or range_end > datetime(2017,1,1,0,0,0) ):
        raise SystemExit("Input date is out of range! Please try again!")
    else:
        print("Correct format and time range!")
        
    if (ask_user == True):
        print("Ready to plot! \n")
        print("Time range: 2016/1/2 - 2016/12/31 (duration included) \n")
        print("Please enter the following format (split by comma): \n")
        print("years, month, day, ploting duration(days) \n")
        print("For example, enter: {}, {}, {}, {}".format(plot_yr, plot_month, plot_day, duration) )
        
        input_format_ok = False
        while(input_format_ok == False):
            user_input = input()
            print("Your input is {}".format(user_input) )
            try:
                plot_yr = int(user_input[0]) 
                plot_month = int(user_input[1]) 
                plot_day = int(user_input[2]) 
                duration = int(user_input[3]) 
                
                range_start = datetime(plot_yr, plot_month, plot_day, 0, 0, 0)
                range_end = datetime(plot_yr, plot_month, plot_day, 0, 0, 0) + relativedelta(days=duration)
                
                if (range_start < datetime(2016,1,2,0,0,0) or range_end > datetime(2017,1,1,0,0,0) ):
                    print("Input date is out of range! Please try again!")
                else:
                    print("Correct format and time range!")
                    input_format_ok = True
            except:
                print("Incorrect format, please try again!")
    
    df_plot = df_plot[range_start.strftime('%Y-%m-%d %H:%M:%S') : range_end.strftime('%Y-%m-%d %H:%M:%S')] 
    # write to csv file
    df_plot_csv_file_name = "df_plot.csv"
    df_plot.to_csv(df_plot_csv_file_name)
    print("Prediction start from {} \n".format(range_start) )
    print("Prediction end at {} \n".format(range_end) )
    print("Detail in {}: \n".format(df_plot_csv_file_name) )
    # print(df_plot)
    # dates = [datetime.fromtimestamp(ts) for ts in df_plot.index ]
    datenums = [ mPlotDATEs.date2num(ts) for ts in df_plot.index ]
    # print(datenums)
    # print(mPlotDATEs.num2date(datenums) )
    # datenums = mPlotDATEs.date2num(dates)
    value_raw = np.array(df_plot['raw_temp_C'])
    value_predict = np.array(df_plot['predict_temp_C'])
    
    plt.figure()
    plt.subplots_adjust(bottom=0.2)
    # plt.xticks( rotation=25 )
    plt.xticks( rotation=60 )
    ax=plt.gca()
    xfmt = mPlotDATEs.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis_date()
    # plt.scatter(y_test.index, y_test)
    # plt.plot(y_test.index, y_predict, color='blue', linewidth=3)
    # plt.scatter(y_test.index[0:25], y_test[0:25])
    # plt.plot(y_test.index[0:25], y_test[0:25], color='red', linewidth=3)
    # plt.plot(y_test.index[0:25], y_predict[0:25], color='blue', linewidth=3)
    # plt.subplot(121)
    plt.xlabel("time range")
    plt.ylabel("degree C")
    plt.title("raw data (red) v.s. predict data (blue)") 
    plt.grid()
    plt.plot(datenums, value_raw, linestyle='-', marker='o', markersize=5, color='r', linewidth=2, label="raw temp C")
    plt.plot(datenums, value_predict, linestyle='-', marker='o', markersize=5, color='b', linewidth=2, label="predict temp C")
    plt.legend(loc="best")
    
    plt.show()
    
    plt.figure()
    # plt.subplot(122)
    plt.xlabel("raw data (degree C)")
    plt.ylabel("predict data (degree C)")
    plt.title("perfect match (red) v.s. model (blue)") 
    plt.grid()
    plt.plot(value_raw, value_raw, linestyle='--', marker='o', markersize=5, color='r', linewidth=1, label="perfect match line")
    plt.scatter(value_raw, value_predict, marker='o', s=10, color='b', label="predict temp C")
    # plt.plot(value_predict, marker='o', markersize=3, color='b', label="predict temp C")
    plt.legend(loc="best")
    
    plt.show()

# poly_degree = int, interaction_only = True
def linear_regr(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, model_result):
    
    # create more features
    poly = preprocessing.PolynomialFeatures(poly_degree, interaction_only=interaction_only)
    
    X_train = poly.fit_transform(X_train) 
    X_test = poly.fit_transform(X_test)
    (s_n, f_n) = X_train.shape
    # l_n = int(math.ceil(1.5*f_n))
    l_n = int(math.ceil(1.2*f_n))
    print("@@@ s_n = {}, f_n = {}, l_n = {}".format(s_n, f_n, l_n) )
    
    np.savetxt("x_train.csv", X_train, delimiter=",")
    np.savetxt("y_train.csv", y_train, delimiter=",")
    np.savetxt("x_test.csv", X_test, delimiter=",")
    np.savetxt("y_test.csv", y_test, delimiter=",")
    
    print("### type of X_train = {}".format(type(X_train)) )
    
    # debug
    for model in [2]:
    # linear regr: [0, 1, 2] NN: [3, 4]
    # for model in [0 1 2 3]:
    # run all: very long runtime
    # for model in [0 1 2 3 4]:
        # model selection
        ## # test score: 0.83
        ## model_name = "SGDRegressor"
        ## model_rt_start = timeit.default_timer()
        ## regr = linear_model.SGDRegressor(penalty='elasticnet', alpha=0.01, l1_ratio=0.25, fit_intercept=True)
        ## model_rt_stop = timeit.default_timer()
        ## model_runtime = model_rt_stop - model_rt_start 
        ## # test score: 0.83
        ## model_name = "ElasticNet"
        ## model_rt_start = timeit.default_timer()
        ## regr = linear_model.ElasticNet(alpha = 0.01)
        ## model_rt_stop = timeit.default_timer()
        ## model_runtime = model_rt_stop - model_rt_start 
        if   (model == 0):
            # test score: 0.84
            alpha = 0 
            model_name = "linear_model.LinearRegression"
            regr = linear_model.LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
            model_rt_start = timeit.default_timer()
            regr.fit(X_train, column_or_1d(y_train) )
            model_rt_stop = timeit.default_timer()
            model_runtime = model_rt_stop - model_rt_start 
            model_result = evaluation(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, 
                            model_result, model_name, model_runtime, regr, alpha)
        elif (model == 1):
            for alpha in [0.0001, 0.001, 0.01, 0.1, 1, 3, 10]:
                # test score: 0.83
                model_name = "linear_model.Lasso"
                regr_lasso = linear_model.Lasso(alpha = alpha)
                model_rt_start = timeit.default_timer()
                regr_lasso.fit(X_train, column_or_1d(y_train) )
                model_rt_stop = timeit.default_timer()
                model_runtime = model_rt_stop - model_rt_start 
                model_result = evaluation(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, 
                                model_result, model_name, model_runtime, regr_lasso, alpha)
        elif (model == 2):
            for alpha in [0.0001, 0.001, 0.01, 0.1, 1, 3, 10]:
            # for alpha in [0.0000001, 0.00001, 0.001, 0.01, 0.1, 1, 3, 10, 30, 100, 300, 10**3, 10**4, 10**5]:
                # test score: 0.84
                model_name = "linear_model.Ridge"
                regr_ridge = linear_model.Ridge(alpha = alpha)
                model_rt_start = timeit.default_timer()
                regr_ridge.fit(X_train, column_or_1d(y_train) )
                model_rt_stop = timeit.default_timer()
                model_runtime = model_rt_stop - model_rt_start 
                model_result = evaluation(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, 
                                model_result, model_name, model_runtime, regr_ridge, alpha)
        elif (model == 3):
            if (poly_degree <= 2):
                for alpha in [0.0001, 0.01, 1]:
                # for alpha in [0.00001]:
                    for layer_n in [3, 7, 11]:
                    # for layer_n in [3]:
                        # test score: 0.83, runtime longer
                        model_name = "neural_network.MLPRegressor, layer = " + str(layer_n)
                        if(layer_n == 3):
                            regr = neural_network.MLPRegressor(random_state=True,hidden_layer_sizes=(l_n,l_n,l_n),alpha=alpha)
                        if(layer_n == 7):
                            regr = neural_network.MLPRegressor(random_state=True,hidden_layer_sizes=(l_n,l_n,l_n,l_n,l_n,l_n,l_n),alpha=alpha)
                        if(layer_n == 11):
                            regr = neural_network.MLPRegressor(random_state=True,hidden_layer_sizes=(l_n,l_n,l_n,l_n,l_n,l_n,l_n,l_n,l_n,l_n,l_n),alpha=alpha)
                        model_rt_start = timeit.default_timer()
                        regr.fit(X_train, column_or_1d(y_train) )
                        model_rt_stop = timeit.default_timer()
                        model_runtime = model_rt_stop - model_rt_start 
                        model_result = evaluation(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, 
                                        model_result, model_name, model_runtime, regr, alpha)
        elif (model == 4):
            if (poly_degree <= 3):
                for alpha in [1, 10, 1000]:
                # for alpha in [0.00001]:
                    # for layer_n in [3, 7, 11]:
                    for layer_n in [7, 11]:
                    # for layer_n in [3]:
                        # test score: 0.83, runtime longer
                        model_name = "neural_network.MLPRegressor, layer = " + str(layer_n)
                        if(layer_n == 3):
                            regr = neural_network.MLPRegressor(random_state=True,hidden_layer_sizes=(l_n,l_n,l_n),alpha=alpha)
                        if(layer_n == 7):
                            # regr = neural_network.MLPRegressor(random_state=True,hidden_layer_sizes=(l_n,l_n,l_n,l_n,l_n,l_n,l_n),alpha=alpha)
                            regr = neural_network.MLPRegressor(random_state=True,hidden_layer_sizes=(l_n,l_n,l_n,l_n,l_n,l_n,l_n),alpha=alpha, learning_rate='invscaling')
                        if(layer_n == 11):
                            regr = neural_network.MLPRegressor(random_state=True,hidden_layer_sizes=(l_n,l_n,l_n,l_n,l_n,l_n,l_n,l_n,l_n,l_n,l_n),alpha=alpha)
                        model_rt_start = timeit.default_timer()
                        regr.fit(X_train, column_or_1d(y_train) )
                        model_rt_stop = timeit.default_timer()
                        model_runtime = model_rt_stop - model_rt_start 
                        model_result = evaluation(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, 
                                        model_result, model_name, model_runtime, regr, alpha)
        else:
            raise SystemExit("Model selection out of range!!!")
        
    return model_result

def evaluation(X_train, y_train, X_test, y_test, poly_degree, interaction_only, print_coef, plot, ask_user, model_result, model_name, model_runtime, regr, alpha):
    print("poly_degree = {}, interaction_only = {}".format(poly_degree, interaction_only))
    with open("logs/log_" + log_timestr +".txt", "a") as logfile:
        logfile.write("====================\n")
        logfile.write("poly_degree = {}, interaction_only = {}\n".format(poly_degree, interaction_only))
    
    print("Model: {} \n".format( model_name ) )
    print("Alpha (Regularization strength): {} \n".format( alpha ) )
    print("X_train.shape = {}".format(X_train.shape) )
    print("y_train.shape = {}".format(y_train.shape) )
    print("X_test.shape = {}".format(X_test.shape) )
    print("y_test.shape = {}".format(y_test.shape) )
    
    if (print_coef):
        # The coefficients
        if hasattr(regr, 'coef_'):
            print("Coefficients: {}\n", regr.coef_)
            with open("logs/log_" + log_timestr +".txt", "a") as logfile:
                logfile.write("Coefficients: {}\n".format(regr.coef_) )
        # for neural_network.MLPRegressor
        if hasattr(regr, 'coefs_'):
            print("Coefficients: {}\n", regr.coefs_)
            with open("logs/log_" + log_timestr +".txt", "a") as logfile:
                logfile.write("Coefficients: {}\n".format(regr.coefs_) )
    
    print("For training set:")
    (mse_train, score_train) = (0, 0)
    # mse_train = float(np.mean( (regr.predict(X_train) - y_train) ** 2) )
    # need to use column_or_1d instead of np.array
    model_rt_predict_train_start = timeit.default_timer()
    predict_train = regr.predict(X_train)
    model_rt_predict_train_stop = timeit.default_timer()
    model_runtime_predict_train = model_rt_predict_train_stop - model_rt_predict_train_start
    mse_train = float( np.mean( (predict_train - column_or_1d(y_train) ) ** 2) )
    score_train = regr.score(X_train, y_train)
    # The mean squared error
    print("Mean squared error (train): {0:.3f} \n".format( mse_train ) )
    # Explained variance score: 1 is perfect prediction
    print("Variance score (train): {0:.3f} \n".format( score_train ) )
    print("model_runtime (training) = {0:.3f} (seconds) \n".format(model_runtime))
    print("model_runtime (predict train set) = {0:.3f} (seconds) \n".format(model_runtime_predict_train))
    
    print("For test set:")
    (mse_test, score_test) = (0, 0)
    model_rt_predict_test_start = timeit.default_timer()
    predict_test = regr.predict(X_test)
    model_rt_predict_test_stop = timeit.default_timer()
    model_runtime_predict_test = model_rt_predict_test_stop - model_rt_predict_test_start
    mse_test = float(np.mean( (predict_test - column_or_1d(y_test) ) ** 2) )
    score_test = regr.score(X_test, y_test)
    # The mean squared error
    print("Mean squared error (test): {0:.3f} \n".format( mse_test ) )
    # Explained variance score: 1 is perfect prediction
    print("Variance score (test): {0:.3f} \n".format( score_test ) )
    print("model_runtime (predict test set) = {0:.3f} (seconds) \n".format(model_runtime_predict_test))
    
    with open("logs/log_" + log_timestr +".txt", "a") as logfile:
        logfile.write("====================\n")
        logfile.write("Features polynomial degree: {} \n".format( poly_degree ) )
        logfile.write("Model: {} \n".format( model_name ) )
        logfile.write("Alpha (Regularization strength): {} \n".format( alpha ) )
        logfile.write("X_train.shape = {} \n".format(X_train.shape) )
        logfile.write("y_train.shape = {} \n".format(y_train.shape) )
        logfile.write("X_test.shape = {} \n".format(X_test.shape) )
        logfile.write("y_test.shape = {} \n".format(y_test.shape) )
        logfile.write("For training set: \n")
        logfile.write("Mean squared error (train): {0:.3f} \n".format( mse_train ) )
        logfile.write("Variance score (train): {0:.3f} \n".format( score_train ) )
        logfile.write("For test set: \n")
        logfile.write("Mean squared error (test): {0:.3f} \n".format( mse_test ) )
        logfile.write("Variance score (test): {0:.3f} \n".format( score_test ) )
        logfile.write("model_runtime (training) = {0:.3f} (seconds) \n".format(model_runtime))
        logfile.write("model_runtime (predict train set) = {0:.3f} (seconds) \n".format(model_runtime_predict_train))
        logfile.write("model_runtime (predict test set) = {0:.3f} (seconds) \n".format(model_runtime_predict_test))
        logfile.write("====================\n")
    
    # collect info.
    (s_n, f_n) = X_train.shape
    model_result.update({(model_name, alpha, int(f_n), poly_degree ):[]})
    ## model_result[(model_name, alpha, int(f_n) )] = ( poly_degree, round(mse_train, 3), round(score_train, 3), 
    ##             round(mse_test, 3), round(score_test, 3), 
    ##             round(model_runtime, 3), round(model_runtime_predict_train, 3), round(model_runtime_predict_test, 3) )
    model_result[(model_name, alpha, int(f_n), poly_degree )] = ( round(mse_train, 3), round(score_train, 3), 
                round(mse_test, 3), round(score_test, 3), 
                round(model_runtime, 3), round(model_runtime_predict_train, 3), round(model_runtime_predict_test, 3) )
    
    # print shape
    if (plot == True):
        plot_y_test(regr, X_test, y_test, ask_user)
    
    return model_result


# def run_fit(postfix, df_run, targets_run, features_run, poly_d_max, inter_only, print_coef, plot):
def run_fit(postfix, df_run_train, df_run_test, targets_run, features_run, poly_d_max, inter_only, print_coef, plot, ask_user):
    text = "RUNNING... df" + postfix
    print("{0:{fill}{align}16}".format(text, fill='=', align='^'))
    (X_train, y_train, X_test, y_test) = (0, 0, 0, 0) 
    # (X_train, y_train, X_test, y_test) = data_gen(df_run, targets_run, features_run, 2006, 2015, 2016, 2016)
    (X_train, y_train, X_test, y_test) = normalization(df_run_train, df_run_test, targets_run, features_run)
    # data = []
    # (data[0], data[1], data[2], data[3]) = data_gen(df_run, features_run)
    print("df{} X_train.shape = {}".format(postfix, X_train.shape))
    print("df{} y_train.shape = {}".format(postfix, y_train.shape))
    print("df{} X_test.shape = {}".format(postfix, X_test.shape))
    print("df{} y_test.shape = {}".format(postfix, y_test.shape))
    print("df_run_train target + features = {}".format(df_run_train.columns.values))
    print("=====")
    
    model_re = {}
    
    # for poly_d in range(1, poly_d_max+1):
    for poly_d in range(1, poly_d_max+1):
        model_re = linear_regr(X_train, y_train, X_test, y_test, poly_degree = poly_d, 
            interaction_only = inter_only, print_coef = print_coef, plot = plot, ask_user = ask_user, model_result = model_re)
    return model_re

def create_new_features(df):
    if ("DATE" in df.columns):
        df["mins_year"] = df["DATE"].apply(lambda x : DateToMinute(x, 'year'))
        df["mins_month"] = df["DATE"].apply(lambda x : DateToMinute(x, 'month'))
        df["mins_day"] = df["DATE"].apply(lambda x : DateToMinute(x, 'day'))
        df["mins_hour"] = df["DATE"].apply(lambda x : DateToMinute(x, 'hour'))
        
        # cos function for seasonal features
        df["cos_mins_year"] = df["mins_year"].apply(lambda x : math.cos( math.radians( (x/(366*24*60))*360 )) )
        df["cos_mins_month"] = df["mins_month"].apply(lambda x : math.cos( math.radians( (x/(31*24*60))*360 )) )
        df["cos_mins_day"] = df["mins_day"].apply(lambda x : math.cos( math.radians( (x/(1*24*60))*360 )) )
    
    if ("HOURLYWindDirection" in df.columns):
        # wind direction normalization
        df["cos_wind_dir"] = df["HOURLYWindDirection"].apply(lambda x : math.cos( math.radians(x) ) )
    
    return df

### end of functions definition ###

st = ['STATION']
targets = ['HOURLYDRYBULBTEMPC']
date = ['DATE']

# removing 
# 'HOURLYVISIBILITY' (no influence) & 
# 'HOURLYSeaLevelPressure' (around 1/10 empty data), no influence
# 'HOURLYPrecip' (no influence & many empty data)
# selective original features
# s_features = [] 
s_features = ['HOURLYDewPointTempC', 'HOURLYRelativeHumidity', 'HOURLYWindSpeed', 'HOURLYWindDirection', 'HOURLYStationPressure']
# s_features = ['HOURLYRelativeHumidity', 'HOURLYWindSpeed', 'HOURLYWindDirection', 'HOURLYStationPressure']
# s_features = ['HOURLYRelativeHumidity', 'HOURLYWindSpeed', 'HOURLYWindDirection']

## # original selected features (excluding targets & date)
## features = ['HOURLYVISIBILITY', 'HOURLYDewPointTempC', 'HOURLYRelativeHumidity', 'HOURLYWindSpeed', 
##             'HOURLYWindDirection', 'HOURLYStationPressure', 'HOURLYSeaLevelPressure', 'HOURLYPrecip']

# df = pd.read_csv('weather_2_stations.csv', parse_dates=['DATE'])
# df = pd.read_csv('weather_2_stations.csv', dtype=date_type, na_values=additional_na_v, parse_dates=['DATE'])

test = ['cos_wind_dir']
features_test_only = test
# create seasonal related features
features_seasonal_only = ['mins_year', 'mins_day', 'cos_mins_year', 'cos_mins_day']
# features_seasonal_only = ['mins_year', 'mins_day', 'cos_wind_dir']
# features_seasonal_only = ['cos_mins_year', 'cos_mins_day', 'cos_wind_dir']
features_seasonal_test = features_seasonal_only + test

features_no_time_no_test = [x for x in s_features if x not in test] 
features_no_test = [x for x in (s_features+features_seasonal_only) if x not in test] 
features_no_time = [x for x in s_features if x not in features_seasonal_only] 

fields = st + targets + date + s_features

## df = pd.read_csv('weather_2_stations.csv', usecols=fields, parse_dates=date)
df = pd.read_csv('station_1.csv', usecols=fields, parse_dates=date)
s1 = df[ df.STATION == 'WBAN:23244' ]
s2 = df[ df.STATION == 'WBAN:23293' ]

# s1.to_csv('s1.csv')
# s2.to_csv('s2.csv')

# station = pd.read_csv('s1.csv', parse_dates=['DATE'])
# station = pd.read_csv('s2.csv', parse_dates=['DATE'])

del df
df = s2[targets + s_features]
df = df.apply(pd.to_numeric, errors='coerce')
df = df.assign( DATE = s2[date])
df = create_new_features(df)

# debug
for experiment in [3]:
# for experiment in [0]:
# for experiment in [0,1,2,3]:

    if   (experiment == 0):
        new_features = targets
    elif (experiment == 1):
        new_features = targets + s_features + features_test_only
    elif (experiment == 2):
        new_features = targets + features_seasonal_only
    elif (experiment == 3):
        new_features = targets + s_features + features_test_only + features_seasonal_only
    else:
        raise sys.SystemExit("new_features is out of range!!!")
    
    print("len(df.index) = {}".format(len(df.index)))
    
    # default training yr
    # HOURLYStationPressure data start at 2005 for station, WBAN:23293
    ## train_yr_start = 2005
    ## station_1.csv 2007.01.01 - 2016.12.31
    train_yr_start = 2007
    train_years = 9 
    test_years = 1
    # train_yr_start = 2006
    # train_years = 10
    # test_years = 1
    test_yr_start = train_yr_start + train_years 
     
    
    # score: 0.68
    # days_later = 365
    # score: 0.69 
    # days_later = 30 
    # score: 0.86
    days_later = 1
    
    # select month, select day
    if(days_later == 365):
        # fixed date
        (s_month, s_day) = (1, 1)
    elif(days_later == 30):
        # s_month: 1 - 11
        (s_month, s_day) = (3, 1)
    elif(days_later == 1):
        # (s_month, s_day) = (6, 1)
        (s_month, s_day) = (1, 1)
    else:
        raise SystemExit("Please enter a valid days_later: 365/30/1.")
    
    new_target = [str(days_later)+"days_later_temp_C"]
    # http://stackoverflow.com/questions/25322933/pandas-timeseries-comparison
    # new_features = targets + features_seasonal_test 
    # df1 = df[date + targets + features_seasonal_test]
    # df2 = df[date + targets + features_seasonal_test]
    # poly_degree = 3, interaction_only = False
    # Mean squared error: 9.63
    # Variance score: 0.66 
    
    df1 = df[date + new_features]
    df2 = df[date + new_features]
    
    df2.loc[:,"DATE"] = df1["DATE"].apply(lambda time_obj: time_obj + relativedelta(days=-days_later))
    df2.rename(columns={str(targets[0]):str(new_target[0])}, inplace=True)
    
    df1 = df1.set_index(["DATE"])
    df2 = df2.set_index(["DATE"])
    # df2 = df2.set_index([new_target])
    
    t1, t2 = df1.align(df2)
    
    t3 = t1
    t3.loc[:, new_target] = t2[new_target]
    # df_time_train = t3['2006':'2014']
    range_start = datetime(train_yr_start, 1, 1, 0, 0, 0)
    range_end   = datetime(train_yr_start, 1, 1, 0, 0, 0) + relativedelta(years=train_years)
    
    df_time_train = t3[range_start.strftime('%Y-%m-%d %H:%M:%S') : range_end.strftime('%Y-%m-%d %H:%M:%S')] 
    print("df_time_train.shape = {}".format(df_time_train.shape))
    df_time_train.loc[:, new_target] = df_time_train[new_target].interpolate(method='time')
    
    # df_time_test = t3['2015'] 
    range_start = datetime(test_yr_start, s_month, s_day, 0, 0, 0)
    range_end   = datetime(test_yr_start, s_month, s_day, 0, 0, 0) + relativedelta(days=365)
    # range_end   = datetime(2017, 3, 05, 0, 0, 0)
    # range_end   = datetime(2017, 3, 05, 0, 0, 0)
    
    df_time_test = t3[range_start.strftime('%Y-%m-%d %H:%M:%S') : range_end.strftime('%Y-%m-%d %H:%M:%S')] 
    print("df_time_test.shape = {}".format(df_time_train.shape))
    
    # drop NaN number rows of test set
    (row_old, col_old) = df_time_test.shape
    print("Before drop NaN number of test set, df_time_test.shape = {}".format(df_time_test.shape))
    df_time_test = df_time_test[ df_time_test.notnull().all(axis=1) ]
    (row, col) = df_time_test.shape
    print("After drop NaN number of test set, df_time_test.shape = {}".format(df_time_test.shape))
    print("Drop rate = {0:.2f} ".format(float(1 - (row/row_old)) ) )
    
    print("===================== \n")
    print("### Experiment = {} \n".format( experiment ) )
    print("new_target = {} \n".format( new_target ) )
    print("new_features = {} \n".format( new_features ) )
    with open("logs/log_" + log_timestr +".txt", "a") as logfile:
        logfile.write("===================== \n")
        logfile.write("### Experiment = {} \n".format( experiment ) )
        logfile.write("new_target = {} \n".format( new_target ) )
        logfile.write("new_features = {} \n".format( new_features ) )
    # run_fit("_predict_", df_time_train, df_time_test, new_target, new_features, poly_d_max=3, inter_only=False, print_coef=False, plot=False, ask_user=False)
    # run_fit("_predict_", df_time_train, df_time_test, new_target, new_features, poly_d_max=2, inter_only=False, print_coef=True, plot=False, ask_user=False)
    # ask_user = False (using default)
    # run_fit("_predict_", df_time_train, df_time_test, new_target, new_features, poly_d_max=1, inter_only=False, print_coef=True, plot=True, ask_user=False)
    # model_re = run_fit("_predict_", df_time_train, df_time_test, new_target, new_features, poly_d_max=3, inter_only=False, print_coef=False, plot=False, ask_user=False)
    # model_re = run_fit("_predict_", df_time_train, df_time_test, new_target, new_features, poly_d_max=2, inter_only=False, print_coef=False, plot=False, ask_user=False)
    model_re = run_fit("_predict_", df_time_train, df_time_test, new_target, new_features, poly_d_max=2, inter_only=False, print_coef=False, plot=True, ask_user=False)
    # ask_user = True (asking user for ploting time range)
    # run_fit("_predict_", df_time_train, df_time_test, new_target, new_features, poly_d_max=2, inter_only=False, print_coef=True, plot=True, ask_user=True)
    # debug 
    # run_fit("_predict_", df_time_train, df_time_test, new_target, new_features, poly_d_max=1, inter_only=False, print_coef=True, plot=False, ask_user=False)
    
    print("### Experiment = {} \n".format( experiment ) )
    # print("model_re = {}".format(model_re))
    
    rt_stop = timeit.default_timer()
    total_runtime = rt_stop - rt_start
    print("runtime = {} (seconds) \n".format(total_runtime))
    with open("logs/log_" + log_timestr +".txt", "a") as logfile:
        logfile.write("total runtime = {} (seconds) \n".format(total_runtime))
        logfile.write("### Experiment = {} \n".format( experiment ) )
        for i in model_re.keys():
            # logfile.write("=== model result === \n")
            logfile.write("model_re[{}] = {}\n".format(i, model_re[i]))
            # logfile.write("{} \n".format(model_re[i]))
            # logfile.write("==================== \n")
