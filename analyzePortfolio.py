import pandas as pd


# Sort by weight in portfolio of the 10 top holdings
def sortByConcentration(dataframe):
    return dataframe.sort_values(by=["Top 10 Holdings Weight (%)"])


# Count how many funds have an open position in a stock
def countOpenPositions(stocks_funds_df):

    count_array = []
    for i in range(len(stocks_funds_df.columns)):
        colname = stocks_funds_df.columns[i]
        counter = stocks_funds_df.iloc[:, i].where(
            stocks_funds_df[colname] != 0).count()
        count_array.append(counter)

    stock_count_df = pd.DataFrame(
        index={"Count"}, columns=stocks_funds_df.columns)
    stock_count_df.iloc[0] = count_array

    stock_count_df = stock_count_df.sort_values(
        by="Count", ascending=False, axis=1)
    return stock_count_df


# Show the biggest agregate positions of the funds
def biggestOpenPositions(stocks_funds_df):
    count_array = []
    for i in range(len(stocks_funds_df.columns)):
        colname = stocks_funds_df.columns[i]
        counter = stocks_funds_df.iloc[:, i].where(
            stocks_funds_df[colname] != 0).sum()
        count_array.append(counter)

    stock_count_df = pd.DataFrame(
        index={"Agregate Percentage"}, columns=stocks_funds_df.columns)
    stock_count_df.iloc[0] = count_array

    stock_count_df = stock_count_df.sort_values(
        by="Agregate Percentage", ascending=False, axis=1)
    return stock_count_df
