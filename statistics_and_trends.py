import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    fig, ax = plt.subplots()
    sns.scatterplot(x='total_bill', y='tip', data=df, ax=ax)
    ax.set_title('Total Bill vs Tip (Relational Plot)')
    plt.savefig('relational_plot.png')
    plt.close(fig)
    return


def plot_categorical_plot(df):
    fig, ax = plt.subplots()
    sns.boxplot(x='day', y='tip', data=df, ax=ax)
    ax.set_title('Tip Amount by Day (Categorical Plot)')
    plt.savefig('categorical_plot.png')
    plt.close(fig)
    return


def plot_statistical_plot(df):
    fig, ax = plt.subplots()
    sns.histplot(df['total_bill'], bins=20, kde=True, ax=ax)
    ax.set_title('Distribution of Total Bill (Statistical Plot)')
    plt.savefig('statistical_plot.png')
    plt.close(fig)
    return


def statistical_analysis(df, col: str):
    mean = df[col].mean()
    stddev = df[col].std()
    skew = ss.skew(df[col], nan_policy='omit')
    excess_kurtosis = ss.kurtosis(df[col], nan_policy='omit')
    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    print("=== Head of Dataset ===")
    print(df.head(), "\n")

    print("=== Descriptive Statistics ===")
    print(df.describe(), "\n")

    print("=== Correlation Matrix ===")
    print(df.select_dtypes(include=[np.number]).corr(), "\n")

    return df


def writing(moments, col):
    mean, stddev, skew, excess_kurtosis = moments
    print(f"For the attribute '{col}':")
    print(
        f"Mean = {mean:.2f}, Standard Deviation = {stddev:.2f}, "
        f"Skewness = {skew:.2f}, and Excess Kurtosis = {excess_kurtosis:.2f}."
    )

    if abs(skew) < 0.5:
        skewness_type = 'approximately symmetric (not skewed)'
    elif skew > 0:
        skewness_type = 'right skewed'
    else:
        skewness_type = 'left skewed'

    if excess_kurtosis > 0.5:
        kurtosis_type = 'leptokurtic (heavy tails)'
    elif excess_kurtosis < -0.5:
        kurtosis_type = 'platykurtic (light tails)'
    else:
        kurtosis_type = 'mesokurtic (normal-like)'

    print(f"The distribution is {skewness_type} and {kurtosis_type}.\n")
    return


def main():
    df = pd.read_csv('data.csv')
    df = preprocessing(df)
    col = 'total_bill'
    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)
    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == '__main__':
    main()
