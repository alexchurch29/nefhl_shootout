import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


def parse_skaters(n=200):
    '''

    parse csv files into pandas dataframes
    :param n: sample size to use for regression
    :return:
    '''
    skaters = pd.read_csv('Players.csv')
    # print(skaters['Att.'].sum())

    league_mean = skaters['Made'].sum() / skaters['Att.'].sum()
    lower_quant = skaters[skaters['Att.']>=10]['Pct.'].quantile(1/3)
    upper_quant = skaters[skaters['Att.']>=10]['Pct.'].quantile(2/3)
    skaters = skaters.drop(columns=['Tm', 'Miss', 'Pct.'])
    skaters = skaters.rename(columns={"Att.": "Attempts", "Made": "Goals"})
    skaters = skaters.groupby('Player').sum()
    skaters['Actual_Conversion_Rate'] = skaters['Goals'] / skaters['Attempts']
    skaters['SOA%'] = skaters['Attempts'] / skaters['Opp.']
    SOA_threshold= skaters['SOA%'].median()
    confidence_intervals = pd.read_csv("confidence_intervals_skaters.csv")
    def map(df):
        lower = confidence_intervals.loc[df['Attempts']][1]
        upper = confidence_intervals.loc[df['Attempts']][2]
        if df['Attempts'] >= n:
            return df['Actual_Conversion_Rate']
        elif df['Attempts'] >= 10 and df['Actual_Conversion_Rate'] > upper:
            return (df['Actual_Conversion_Rate'] * (df['Attempts'] / n) + upper_quant * ((n - df['Attempts'])/ n))
        elif df['SOA%'] < SOA_threshold or df['Attempts'] < 4 or (df['Attempts'] >=10 and df['Actual_Conversion_Rate'] < lower):
            return (df['Actual_Conversion_Rate'] * (df['Attempts'] / n) + lower_quant * ((n - df['Attempts'])/ n))
        else:
            return (df['Actual_Conversion_Rate'] * (df['Attempts'] / n) + league_mean * ((n - df['Attempts'])/ n))
    skaters['Estimated_Conversion_Rate'] = skaters.apply(map, axis = 1)
    skaters = skaters.sort_values(by=['Estimated_Conversion_Rate'], ascending=False)
    max_val = skaters['Estimated_Conversion_Rate'].max()
    min_val = skaters['Estimated_Conversion_Rate'].min()
    spread = (max_val - min_val) * 100
    skaters['SO_Rating'] = 50 + (((skaters['Estimated_Conversion_Rate'] - min_val) * 100) * (49 / spread))

    skaters['Estimated_Conversion_Rate1'] = (skaters['Estimated_Conversion_Rate'] * (1 - league_mean)) / (1 - skaters['Estimated_Conversion_Rate'])

    # skaters1 = skaters[['Opp.', 'Attempts', 'SOA%', 'Goals', 'Actual_Conversion_Rate', 'Estimated_Conversion_Rate', 'SO_Rating']]
    # skaters1.to_csv('skater_ratings.csv')
    skaters = skaters.drop(columns=['SOA%', 'Opp.'])
    # skaters.to_csv('skaters_test_n_equalto_' + str(n) + '.csv')

    return skaters


def parse_goalies(n=200):
    '''
    parse csv files into pandas dataframes
    :param n: sample size to use for regression
    :return:
    '''
    goalies = pd.read_csv('goalies_raw_data.csv')
    # print(goalies['Att.'].sum())

    league_mean = goalies['Miss'].sum() / goalies['Att.'].sum()
    lower_quant = goalies[goalies['Att.']>=10]['Pct.'].quantile(1/3) / 100
    goalies = goalies.drop(columns=['Team', 'Made', 'Pct.'])
    goalies = goalies.rename(columns={"Att.": "Attempts", "Miss": "Saves"})
    goalies = goalies.groupby('Player').sum()
    goalies['Actual_Save_Rate'] = goalies['Saves'] / goalies['Attempts']
    # confidence_intervals = pd.read_csv("confidence_intervals_goalies.csv")

    def map(df):
        if df['Attempts'] >= n:
            return df['Actual_Save_Rate']
        elif df['Attempts'] < 10:
            return (df['Actual_Save_Rate'] * (df['Attempts'] / n) + lower_quant * ((n - df['Attempts'])/ n))
        else:
            return (df['Actual_Save_Rate'] * (df['Attempts'] / n) + league_mean * ((n - df['Attempts'])/ n))
    goalies['Estimated_Save_Rate'] = goalies.apply(map, axis=1)
    goalies = goalies.sort_values(by=['Estimated_Save_Rate'], ascending=False)
    max_val = goalies['Estimated_Save_Rate'].max()
    min_val = goalies['Estimated_Save_Rate'].min()
    spread = (max_val - min_val) * 100
    goalies['SO_Rating'] = 50 + (((goalies['Estimated_Save_Rate'] - min_val) * 100) * (49 / spread))

    goalies['Estimated_Save_Rate1'] = (goalies['Estimated_Save_Rate'] * (1 - league_mean)) / (1 - goalies['Estimated_Save_Rate'])

    # goalies.to_csv('goalies_test_n_equalto_' + str(n) + '.csv')
    # skaters1 = goalies[['Attempts', 'Saves', 'Actual_Save_Rate', 'Estimated_Save_Rate', 'SO_Rating']]
    # skaters1.to_csv('goalie_ratings.csv')

    return goalies


def simulate(type=0, n=10000):
    '''
    simulates individual shootout matchups using a weighted random sample
    :return:
    '''
    skaters = pd.read_csv("skaters_test_n_equalto_200.csv")
    goalies = pd.read_csv("goalies_test_n_equalto_200.csv")

    skaters = pd.DataFrame(
        [skaters.loc[idx] for idx in skaters.index for _ in range(int(skaters.loc[idx]['Attempts']))]).reset_index(
        drop=True)
    goalies = pd.DataFrame(
        [goalies.loc[idx] for idx in goalies.index for _ in range(int(goalies.loc[idx]['Attempts']))]).reset_index(
        drop=True)

    s = skaters.shape[0]
    g = goalies.shape[0]

    if type == 0:

        goals = 0
        saves = 0

        for i in range(n):
            # print('\n')
            r1 = random.randint(0, s - 1)
            r2 = random.randint(0, g - 1)
            skater = skaters.iloc[[r1]]
            goalie = goalies.iloc[[r2]]
            # print(skaters.iloc[r1])
            # print(goalies.iloc[r2])
            # print('xG: ' + str(skaters.iloc[r1][5] / (goalies.iloc[r2][5] + skaters.iloc[r1][5])))
            p1 = int(skaters.iloc[r1][6] * 100)
            p2 = int(goalies.iloc[r2][6] * 100)
            r3 = random.randint(1, p1 + p2)
            if r3 <= p1:
                # print("Result: Goal")
                goals += 1
                # print('\n')
            else:
                # print("Result: No Goal")
                saves += 1
                # print('\n')
        print(goals / (saves + goals))

    elif type == 1:

        goals = 0
        saves = 0

        r1 = random.randint(0, s - 1)
        print(skaters.iloc[r1])

        for i in range(n):
            # print('\n')
            r2 = random.randint(0, g - 1)
            skater = skaters.iloc[[r1]]
            goalie = goalies.iloc[[r2]]
            # print(goalies.iloc[r2])
            # print('xG: ' + str(skaters.iloc[r1][3] / (goalies.iloc[r2][3] + skaters.iloc[r1][3])))
            p1 = int(skaters.iloc[r1][6] * 100)
            p2 = int(goalies.iloc[r2][6] * 100)
            r3 = random.randint(1, p1 + p2)
            if r3 <= p1:
                # print("Result: Goal")
                goals += 1
            else:
                # print("Result: No Goal")
                saves += 1
                # print('\n')
        print(goals / (saves + goals))

    else:

        goals = 0
        saves = 0

        r2 = random.randint(0, g - 1)
        print(goalies.iloc[r2])

        for i in range(n):
            # print('\n')
            r1 = random.randint(0, s - 1)
            skater = skaters.iloc[[r1]]
            goalie = goalies.iloc[[r2]]
            # print(skaters.iloc[r1])
            # print('xG: ' + str(skaters.iloc[r1][3] / (goalies.iloc[r2][3] + skaters.iloc[r1][3])))
            p1 = int(skaters.iloc[r1][6] * 100)
            p2 = int(goalies.iloc[r2][6] * 100)
            r3 = random.randint(1, p1 + p2)
            if r3 <= p1:
                # print("Result: Goal")
                goals += 1
            else:
                # print("Result: No Goal")
                saves += 1
                # print('\n')
        print(saves / (saves + goals))


def confidence_interval_skaters(name="Frans Nielsen", n=1000, avg=True, att = 50, confidence=0.95):
    '''
    :return:
    '''
    goalies = pd.read_csv('goalies_test_n_equalto_200.csv')
    goalies = pd.DataFrame(
        [goalies.loc[idx] for idx in goalies.index for _ in range(int(goalies.loc[idx]['Attempts']))]).reset_index(
        drop=True)
    g = goalies.shape[0]

    skaters = parse_skaters(n=att)
    league_mean = skaters['Goals'].sum() / skaters['Attempts'].sum()
    p = (league_mean * (1 - league_mean)) / (1 - league_mean)
    skater = skaters.loc[name]
    if att == -1:
        att = skater['Attempts']
    distribution = []

    for i in range(n):
        goals = 0
        saves = 0
        for i in range(int(att)):
            r2 = random.randint(0, g - 1)
            if not avg:
                p1 = int(skater[5] * 100)
            else:
                p1 = int(p * 100)
            p2 = int(goalies.iloc[r2][6] * 100)
            r3 = random.randint(1, p1 + p2)
            if r3 <= p1:
                goals += 1
            else:
                saves += 1
        distribution.append(goals / (saves + goals))

    values, bins, _ = plt.hist(distribution, normed=True)

    def mean_confidence_interval(data, confidence=confidence):
        import scipy.stats as stats
        a = np.array(data)
        mean, sigma = a.mean(), a.std(ddof=1)
        conf_int_a = stats.norm.interval(confidence, loc=mean, scale=sigma)
        return conf_int_a

    def find_bin_idx_of_value(bins, value):
        """Finds the bin which the value corresponds to."""
        array = np.asarray(value)
        idx = np.digitize(array, bins)
        return idx - 1

    def area_after_val(counts, bins, val):
        """Calculates the area of the hist after a certain value"""
        left_bin_edge_index = find_bin_idx_of_value(bins, val)
        bin_width = np.diff(bins)[0]
        area = sum(bin_width * counts[left_bin_edge_index:])
        return area

    # print(area_after_val(values, bins, 7 / 23))

    ci = mean_confidence_interval(distribution)
    # print(str(ci[0]) + ", " + str(ci[1]))
    # plt.show()

    return ci[0], ci[1]


def confidence_interval_goalies(name, n=1000, avg=True, att = 50, confidence=0.95):
    '''
    :return:
    '''
    skaters = pd.read_csv('skaters_test_n_equalto_200.csv')
    skaters = pd.DataFrame(
        [skaters.loc[idx] for idx in skaters.index for _ in range(int(skaters.loc[idx]['Attempts']))]).reset_index(
        drop=True)
    g = skaters.shape[0]

    goalies = parse_goalies(n=att)
    league_mean = skaters['Goals'].sum() / skaters['Attempts'].sum()
    p = ((1 - league_mean) * league_mean) / (league_mean)
    goalie = goalies.loc[name]
    if att == -1:
        att = goalie['Attempts']
    distribution = []

    for i in range(n):
        goals = 0
        saves = 0
        for i in range(int(att)):
            r2 = random.randint(0, g - 1)
            if not avg:
                p1 = int(goalie[5] * 100)
            else:
                p1 = int(p * 100)
            p2 = int(skaters.iloc[r2][6] * 100)
            r3 = random.randint(1, p1 + p2)
            if r3 <= p1:
                saves += 1
            else:
                goals += 1
        distribution.append(saves / (saves + goals))

    values, bins, _ = plt.hist(distribution, normed=True)

    def mean_confidence_interval(data, confidence=confidence):
        import scipy.stats as stats
        a = np.array(data)
        mean, sigma = a.mean(), a.std(ddof=1)
        conf_int_a = stats.norm.interval(confidence, loc=mean, scale=sigma)
        return conf_int_a

    ci = mean_confidence_interval(distribution)
    # print(str(ci[0]) + ", " + str(ci[1]))
    # plt.show()

    return ci[0], ci[1]


if __name__ == '__main__':
    parse_skaters()
    # parse_goalies()
    # simulate(type=2)
    # df = pd.DataFrame(columns=('Lower', 'Upper'))
    # for i in range(4,6):
    #     print(i)
    #     lower, upper = confidence_interval_skaters(att=i)
    #     df.loc[i] = [lower, upper]
    # df.to_csv("confidence_intervals_skaters.csv")