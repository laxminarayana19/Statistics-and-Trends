"""
This is the template file for the statistics and trends assignment.
You will be expected to complete all the sections and
make this a fully working, documented file.
You should NOT change any function, file or variable names,
if they are given to you here.
Make use of the functions presented in the lectures
and ensure your code is PEP-8 compliant, including docstrings.
"""
from corner import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns

def plot_relational_plot(df):
    """
    Plots a relational scatterplot between total_bill and tip.
    """
    fig, ax = plt.subplots()
    sns.scatterplot(x='total_bill', y='tip', data=df, ax=ax)
    ax.set_title('Total Bill vs Tip (Relational Plot)')
    plt.savefig('relational_plot.png')
    plt.close(fig)
    return

def plot_categorical_plot(df):
    """
    Plots a categorical boxplot for tip amount by day.
    """
    fig, ax = plt.subplots()
    sns.boxplot(x='day', y='tip', data=df, ax=ax)
    ax.set_title('Tip Amount by Day (Categorical Plot)')
    plt.savefig('categorical_plot.png')
    plt.close(fig)
    return

def plot_statistical_plot(df):
    """
    Plots a histogram for total_bill.
    """
    fig, ax = plt.subplots()
    sns.histplot(df['total_bill'], bins=20, kde=True, ax=ax)
    ax.set_title('Distribution of Total Bill (Statistical Plot)')
    plt.savefig('statistical_plot.png')
    plt.close(fig)
    return

def statistical_analysis(df, col: str):
    """
    Computes mean, standard deviation, skewness, and excess kurtosis for the given column.
    """
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col], nan_policy='omit')
    excess_kurtosis = ss.kurtosis(df[col], nan_policy='omit')
    return mean, stddev, skew, excess_kurtosis

def preprocessing(df):
    """Preprocesses the dataset, provides quick stats and returns the cleaned dataframe."""
    print(df.head())
    print(df.describe())
    print(df.select_dtypes(include=[np.number]).corr())
    # No complex cleaning needed for tips dataset, but placeholder for future.
    return df

def writing(moments, col):
    """
    Prints the calculated statistics and interprets skew/kurtosis.
    """
    print(f'For the attribute {col}:')
    print(f'Mean = {moments[0]:.2f}, '
          f'Standard Deviation = {moments[1]:.2f}, '
          f'Skewness = {moments[2]:.2f}, and '
          f'Excess Kurtosis = {moments[3]:.2f}.')

    if abs(moments[2]) < 0.5:
        skewness_type = 'not skewed'
    elif moments[2] > 0:
        skewness_type = 'right skewed'
    else:
        skewness_type = 'left skewed'

    if moments[3] > 0.5:
        kurtosis_type = 'leptokurtic'
    elif moments[3] < -0.5:
        kurtosis_type = 'platykurtic'
    else:
        kurtosis_type = 'mesokurtic'
    return

def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'total_bill'  # Example analysis column from tips dataset
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return

if __name__ == '__main__':
    main()
