import pandas as pd
import numpy as np

def parse_skaters(n=20):
    '''
    parse csv files into pandas dataframes
    :param n: sample size to use for regression
    :return:
    '''
    skaters = pd.read_csv('skaters.csv')
    # print(skaters['Att.'].sum())

    league_median = skaters['Pct.'].median() / 100
    skaters = skaters.drop(columns=['Tm', 'Miss', 'Pct.'])
    skaters = skaters.rename(columns={"Att.": "Attempts", "Made": "Goals"})
    skaters = skaters.groupby('Player').sum()
    skaters['Actual_Conversion_Rate'] = skaters['Goals'] / skaters['Attempts']
    # league_avg = skaters['Goals'].sum() / skaters['Attempts'].sum()
    skaters['Estimated_Conversion_Rate'] = np.where(skaters['Attempts'] >= n, skaters['Actual_Conversion_Rate'],
                                                    (skaters['Actual_Conversion_Rate'] * (skaters['Attempts'] / n) +
                                                      league_median * ((n - skaters['Attempts'])/ n)))
    skaters = skaters.sort_values(by=['Estimated_Conversion_Rate'], ascending=False)
    skaters['SO_Rating'] = 50 + 50 * skaters['Estimated_Conversion_Rate']

    # print(skaters)
    skaters.to_csv('skaters_test.csv')
    return


def parse_goalies(n=80):
    '''
    parse csv files into pandas dataframes
    :param n: sample size to use for regression
    :return:
    '''
    goalies = pd.read_csv('goalies.csv')
    # print(goalies['Att.'].sum())

    league_median = goalies['Pct.'].median() / 100
    goalies = goalies.drop(columns=['Team', 'Made', 'Pct.'])
    goalies = goalies.rename(columns={"Att.": "Attempts", "Miss": "Saves"})
    goalies = goalies.groupby('Player').sum()
    goalies['Actual_Save_Rate'] = goalies['Saves'] / goalies['Attempts']
    # league_avg = goalies['Saves'].sum() / goalies['Attempts'].sum()
    goalies['Estimated_Save_Rate'] = np.where(goalies['Attempts'] >= n, goalies['Actual_Save_Rate'],
                                                    (goalies['Actual_Save_Rate'] * (goalies['Attempts'] / n) +
                                                      league_median * ((n - goalies['Attempts'])/ n)))
    goalies = goalies.sort_values(by=['Estimated_Save_Rate'], ascending=False)
    goalies['SO_Rating'] = 50 + 50 * goalies['Estimated_Save_Rate']

    # print(goalies)
    goalies.to_csv('goalies_test.csv')
    return 


if __name__ == '__main__':
    parse_skaters()
    parse_goalies()