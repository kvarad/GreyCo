import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

def question_2a():
    #return the zip corresponding to the highest average loan amount
    print df.groupby("Zip")["Loan Amount"].mean().idxmax()
    '''
    1a. This returns a zipcode of 19020, which is in Pennsylvania

    '''

def question_2b():
    #get rid of irregular property values
    df["Property Value"].replace("Error", np.NaN, inplace=True)
    df["Property Value"].replace("0.0", np.NaN, inplace=True)

    #create new column which contains the % of taxes to PV, convert cols to float data types
    df["Percentage of tax to PV"] = (((df["Taxes Expense"].astype(float))/(df["Property Value"]).astype(float)) * 100)

    #get the record corresponding to the highest % of tax to PV
    print df.loc[df['Percentage of tax to PV'].idxmax()]

    '''
    1b. This returns the zipcode 7017, which is East Orange, New Jersey.
    '''

def question_2c():
    print df.corr(method="pearson")["Maintenance Expense"]
    print df.corr(method="spearman")["Maintenance Expense"]

    '''
    1c. The correlation between "Total Operating Expenses" seems to have the strongest
        correlation to "Maintenance Expense". This suggests that Maintenance Expenses
        probably account for a significant chunk of the total operating cost of the
        property.
    '''

def question_2d():
    #scrub the data to remove irrgular fields
    df["Property Value"].replace("Error", np.NaN, inplace=True)
    df["Property Value"].replace("0.0", np.NaN, inplace=True)
    df["Loan Amount"].replace(0.0, np.NaN, inplace=True)

    #create LTV column using property value and loan amount columns
    df["LTV"] = (((df["Loan Amount"].astype(float))/(df["Property Value"]).astype(float)))

    print df[["Property Value", "Loan Amount", "LTV"]]
    print "Median: %s" %df.median()["LTV"]
    print "Range: %s" %(df.max()["LTV"] - df.min()["LTV"])
    print "Variance: %s" %df.var()["LTV"]

    '''
    1d. It appears that more than half of home loans are for more than 50
        percent of the property value.
    '''

def question_2e():
    #scrub the data
    df["Property Value"].replace("Error", np.NaN, inplace=True)
    df["Property Value"].replace("0.0", np.NaN, inplace=True)
    df["Property Value"] = pd.to_numeric(df["Property Value"], errors='coerce')

    #create sample of 100 largest and smallest valuations
    df_highs = df.nlargest(100, "Property Value")
    df_lows = df.nsmallest(100, "Property Value")

    #find correlation b/w properties and their expenses
    print df_highs[[
                    "Property Value",
                    "Total Operating Expenses",
                     "Maintenance Expense",
                     "Parking Expense",
                     "Taxes Expense",
                     "Insurance Expense",
                     "Utilities Expense",
                     "Payroll Expense"
                     ]].corr()["Property Value"]

    print df_lows[[
                     "Property Value",
                     "Total Operating Expenses",
                     "Maintenance Expense",
                     "Parking Expense",
                     "Taxes Expense",
                     "Insurance Expense",
                     "Utilities Expense",
                     "Payroll Expense"
                     ]].corr()["Property Value"]

    '''
    1e. It would appear that larger valuations do have the advantage of saving on
    utilities expenses, as there exists a positive correlation between property
    value and utily costs, but overall there does not appear to be too great a
    savings related to economies of scale for other expenses. Generally speaking,
    expenses seem to rise as the valuation of a property becomes larger. As the
    adage goes, "Mo' money, mo' problems."
    '''

def question_3a():
    df["Total Operating Expenses"].replace("0.0", np.NaN, inplace=True)
    df["Effective Gross Income"].replace("0.0", np.NaN, inplace=True)

    df["Expense Ratio"] = (((df["Total Operating Expenses"].astype(float))/(df["Effective Gross Income"]).astype(float)))
    df[["Expense Ratio"]].plot()
    plt.show()

def question_3b():
    #convert year built column to datetime and extract the year value
    df["Total Operating Expenses"].replace("0.0", np.NaN, inplace=True)
    df["Year Built"] = pd.to_datetime(df["Year Built"], errors="coerce", format="%Y")
    current_date = datetime.datetime.now()

    #calculate the age of the property to find correlation to operating costs
    df["Property Age"] = current_date.year - (df["Year Built"].apply(lambda x: x.year))

    #find correlation and pyplot
    print df[["Property Age", "Total Operating Expenses"]].corr()["Property Age"].plot()
    plt.show()

def question_3c(df):
    df["Loan Amount"] = pd.to_numeric(df["Loan Amount"], errors='coerce')
    df["Loan Amount"].replace(0.0, np.NaN, inplace=True)

    df["First Payment Date"] = pd.to_datetime(df["First Payment Date"], errors="coerce", infer_datetime_format=True)
    df["Maturity Date"] = pd.to_datetime(df["Maturity Date"], errors="coerce", infer_datetime_format=True)

    df2 = pd.DataFrame()
    df2["Fiscal Quarter"] = pd.period_range(df.min()["First Payment Date"],datetime.datetime.today(), freq="Q-Dec")

    df2["Loan Amount"] = df["Loan Amount"]
    print df2
    LA_sum = {"Loan Amount": np.sum}

    result = df2.groupby("Fiscal Quarter")["Loan Amount"].agg(LA_sum)

    cumsum = result.groupby(level="Fiscal Quarter")["Loan Amount"].transform(lambda x: x.cumsum())
    c = cumsum.plot()
    plt.show()

def question4(df):
    '''
    dropping date columns from dataset as they're causing issues regression
    consider converting them using 1-of-K encoding via DictVectorization

    Also dropping Zip bc there's 12 fewer records in the df than for all other
    columns. This becomes a nightmare when attempting to fit a regression model
    to the dataframe later on.
    '''
    df.drop(["First Payment Date", "Maturity Date", "Year Built", "Zip"],axis=1, inplace=True)

    #scrub the dataset of irregular values for Loan Amount, Property Value and Year Built
    df = df[df["Loan Amount"] != 0]
    df = df[df["Property Value"] != 0]
    df = df[df["Property Value"] != "Error"]

    df = df[pd.notnull(df)]
    df = df.apply(lambda x: pd.to_numeric(x,errors='ignore'))

    df["LTV"] = df["Loan Amount"]/df["Property Value"]
    df2 = pd.DataFrame({"LTV":df["LTV"]})
    df.drop(["LTV"], axis=1, inplace=True)
    print df2


    #create training and testing datasets
    X_train, X_test, Y_train, Y_test = train_test_split(
        df,
        df2,
        random_state=5
    )

    lin_reg = LinearRegression(fit_intercept=True, normalize=True)
    lin_reg.fit(X_train, Y_train)

    pd.DataFrame(zip(df.columns, lin_reg.coef_), columns = ['features', 'estimated_coefficients'])
    '''
    At this point I would infer that due to the date data being left out, the correlation is not
    as robust as it could be for all the available data. The model can be improved by including all
    available data as well as supplying more data points to make references to.
    '''

if __name__ == "__main__":
    df = pd.read_csv("~/Desktop/Code/GreyStone/re_data.csv")
    #question_2a()
    #question_2b()
    #question_2c()
    #question_2d()
    #question_2e()
    #question_3a()
    #question_3b()
    #question_3c(df)
    question4(df)
