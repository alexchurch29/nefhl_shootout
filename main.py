import pandas as pd
import numpy as np
import random


def parse_skaters(n=30):
    '''
    parse csv files into pandas dataframes
    :param n: sample size to use for regression
    :return:
    '''
    skaters = pd.read_csv('skaters_raw_data.csv')
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
    max_val = skaters['Estimated_Conversion_Rate'].max()
    min_val = skaters['Estimated_Conversion_Rate'].min()
    spread = (max_val - min_val) * 100
    skaters['SO_Rating'] = 50 + (((skaters['Estimated_Conversion_Rate'] - min_val) * 100) * (49 / spread))
    print(skaters['SO_Rating'].quantile(0.25))

    league_mean = skaters['Goals'].sum() / skaters['Attempts'].sum()
    print(league_mean)

    skaters['Estimated_Conversion_Rate1'] = (skaters['Estimated_Conversion_Rate'] * (1 - league_mean)) / (1 - skaters['Estimated_Conversion_Rate'])

    # print(skaters)
    # skaters.to_csv('skaters_test.csv')
    return skaters


def parse_goalies(n=50):
    '''
    parse csv files into pandas dataframes
    :param n: sample size to use for regression
    :return:
    '''
    goalies = pd.read_csv('goalies_raw_data.csv')
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
    max_val = goalies['Estimated_Save_Rate'].max()
    min_val = goalies['Estimated_Save_Rate'].min()
    spread = (max_val - min_val) * 100
    goalies['SO_Rating'] = 50 + (((goalies['Estimated_Save_Rate'] - min_val) * 100) * (49 / spread))
    print(goalies['SO_Rating'].quantile(0.25))

    skaters = pd.read_csv('skaters.csv')
    league_mean = skaters['Made'].sum() / skaters['Att.'].sum()
    goalies['Estimated_Save_Rate1'] = (goalies['Estimated_Save_Rate'] * (league_mean)) / (1 - goalies['Estimated_Save_Rate'])

    # print(goalies)
    # goalies.to_csv('goalies_test.csv')
    return goalies


def simulate():
    '''
    simulates individual shootout matchups using a weighted random sample
    :return:
    '''
    skaters = parse_skaters()
    goalies = parse_goalies()

    skaters = pd.DataFrame(
        [skaters.ix[idx] for idx in skaters.index for _ in range(int(skaters.ix[idx]['Attempts']))]).reset_index(
        drop=True)
    goalies = pd.DataFrame(
        [goalies.ix[idx] for idx in goalies.index for _ in range(int(goalies.ix[idx]['Attempts']))]).reset_index(
        drop=True)

    s = skaters.shape[0]
    g = goalies.shape[0]

    goals = 0
    saves = 0

    for i in range(1000000):
        # print('\n')
        r1 = random.randint(0, s - 1)
        r2 = random.randint(0, g - 1)
        skater = skaters.iloc[[r1]]
        goalie = goalies.iloc[[r2]]
        print(skaters.iloc[r1])
        print(goalies.iloc[r2])
        print('xG: ' + str(skaters.iloc[r1][5] / (goalies.iloc[r2][5] + skaters.iloc[r1][5])))
        p1 = int(skaters.iloc[r1][5] * 100)
        p2 = int(goalies.iloc[r2][5] * 100)
        r3 = random.randint(1, p1 + p2)
        if r3 <= p1:
            print("Result: Goal")
            goals += 1
        else:
            print("Result: No Goal")
            saves += 1
            print('\n')
    print(goals / (saves + goals))
    #
    # goals = 0
    # saves = 0
    #
    # r1 = random.randint(0, s - 1)
    # print(skaters.iloc[r1])
    #
    # for i in range(10000):
    #     # print('\n')
    #     r2 = random.randint(0, g - 1)
    #     skater = skaters.iloc[[r1]]
    #     goalie = goalies.iloc[[r2]]
    #     # print(goalies.iloc[r2])
    #     # print('xG: ' + str(skaters.iloc[r1][3] / (goalies.iloc[r2][3] + skaters.iloc[r1][3])))
    #     p1 = int(skaters.iloc[r1][5] * 100)
    #     p2 = int(goalies.iloc[r2][5] * 100)
    #     r3 = random.randint(1, p1 + p2)
    #     if r3 <= p1:
    #         # print("Result: Goal")
    #         goals += 1
    #     else:
    #         # print("Result: No Goal")
    #         saves += 1
    #         # print('\n')
    # print(goals / (saves + goals))
    #
    # goals = 0
    # saves = 0
    #
    # r2 = random.randint(0, g - 1)
    # print(goalies.iloc[r2])
    #
    # for i in range(10000):
    #     # print('\n')
    #     r1 = random.randint(0, s - 1)
    #     skater = skaters.iloc[[r1]]
    #     goalie = goalies.iloc[[r2]]
    #     # print(skaters.iloc[r1])
    #     # print('xG: ' + str(skaters.iloc[r1][3] / (goalies.iloc[r2][3] + skaters.iloc[r1][3])))
    #     p1 = int(skaters.iloc[r1][5] * 100)
    #     p2 = int(goalies.iloc[r2][5] * 100)
    #     r3 = random.randint(1, p1 + p2)
    #     if r3 <= p1:
    #         # print("Result: Goal")
    #         goals += 1
    #     else:
    #         # print("Result: No Goal")
    #         saves += 1
    #         # print('\n')
    # print(saves / (saves + goals))


if __name__ == '__main__':
    # parse_skaters()
    # parse_goalies()
    simulate()
