# 230p_cnn_project

#Task 1 (50 points):

In this exercise, we test to what extent a model trained to detect BB patterns can capture opportunities with large short term gains. Get AAPL stock adjusted closing prices from 2010-01-01 to 2018-12-31.

Select a time window \{5,10,20,30\} days. Generate stock graphs sequentially using just the adjusted closing price column and label the graphs using BB of 20 periods and 1 std. Label the graph buy if the current adjusted closing price is below the lower BB bound and sell is it is higher than the upper bound. Drop the rest. The exact size and resolution of the figures are hyperparameters you can choose, but the graphs should only include adjusted close prices and relevant features engineered based on these prices.

Using the same dataset and time window, create the same graphs but label them based on their short term changes:

If the stock rises by at least 2\% in the next 5 days, label the graph as buy; If it drops by at least 2\%, label it as sell. Drop the rest.


In this problem, please use 123 as the random seed for splitting and optimization.


1) Train a CNN with a 80-10-10 train-validate split using the graphs labeled by BB. Report the accuracy and F1 score. F1 score is the primary performance measure. (20 pts)

2) Train another CNN with a 80-10-10 train-validate split using the graphs labeled by short term gains. Report the accuracy and F1 score. F1 score is the primary performance measure. (25 pts)

Hint: consider dropout layers or regularization to prevent overfitting.

Hint: A bollinger bands (BB) is a pair of upper and lower bounds given a period length and band width. For example, a 20 period 1 std BB has an upper bound of 20-day moving average + 1 std of 20 days and a lower bound of 20-day moving average - 1 std of 20 days.


3) Discuss why designing models using a random 80-10-10 split in stock forecasting is problematic. (5 pts)


4) (Optional tiebreaker -- this part is only counted if there is a tie between multiple groups) Use the following scheme to test which strategy will perform to best over the next 5 days, 10 days, 1 month, and 6 months (starting from 2019-01-01).

Compare three strategies in total: buy-sell decisions produced by the BB CNN model, short term gain CNN model, and BB without CNN (buy if current adjusted closing is below the lower bound and sell if it is higher than the upper).

Every time a buy signal occurs, buy 1 share of the stock at the current adjusted closing price. Every time a sell signal occurs, sell 1 share of the stock at the current closing price. At the end of the each time interval, calculate the net gain following:

$$\text{net gain}=\text{gain on selling} + \text{value of current holding} - \text{spending on purchasing}$$

Try to develop a strategy using CNN or BB or a mix of the two to maximize the net gain over the next month. You can vary the quantity of each trade, but each trade cannot be more than 5 share of the stock. Report the strategy and the net gain.

#Task 2 (10 pts):

Using the same AAPL data set in task 1. The test performance will be evaluated in a period of a (undisclosed) week within the first 3 months of 2019. Report the MSE. Select any time window in the set \{5,10,20,30\} days. Note that the size of the time window may affect the performance of your model. Upload the saved model and a code that takes in a yahoo finance dataframe and reports the MSE based on the model. Make sure the code can load the model correctly (I can adjust the file path if needed, but otherwise the code should run without any modification).
"""
