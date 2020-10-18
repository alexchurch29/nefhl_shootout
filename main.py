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
    league_mean = skaters['Goals'].sum() / skaters['Attempts'].sum()
    skaters['Estimated_Conversion_Rate'] = skaters['Estimated_Conversion_Rate'] * (league_mean / league_median)
    skaters = skaters.sort_values(by=['Estimated_Conversion_Rate'], ascending=False)
    max_val = skaters['Estimated_Conversion_Rate'].max()
    min_val = skaters['Estimated_Conversion_Rate'].min()
    spread = (max_val - min_val) * 100
    skaters['SO_Rating'] = 50 + (((skaters['Estimated_Conversion_Rate'] - min_val) * 100) * (49 / spread))

    skaters['Estimated_Conversion_Rate1'] = (skaters['Estimated_Conversion_Rate'] * (1 - league_mean)) / (1 - skaters['Estimated_Conversion_Rate'])

    # print(skaters)
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

    skaters = pd.read_csv('skaters_raw_data.csv')
    league_mean = skaters['Made'].sum() / skaters['Att.'].sum()
    goalies['Estimated_Save_Rate1'] = (goalies['Estimated_Save_Rate'] * (league_mean)) / (1 - goalies['Estimated_Save_Rate'])

    # print(goalies)
    # goalies.to_csv('goalies_test_n_equalto_' + str(n) + '.csv')
    return goalies


def simulate(type=0, n=10000):
    '''
    simulates individual shootout matchups using a weighted random sample
    :return:
    '''
    skaters = parse_skaters()
    goalies = parse_goalies()

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
            p1 = int(skaters.iloc[r1][5] * 100)
            p2 = int(goalies.iloc[r2][5] * 100)
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
            p1 = int(skaters.iloc[r1][5] * 100)
            p2 = int(goalies.iloc[r2][5] * 100)
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
            p1 = int(skaters.iloc[r1][5] * 100)
            p2 = int(goalies.iloc[r2][5] * 100)
            r3 = random.randint(1, p1 + p2)
            if r3 <= p1:
                # print("Result: Goal")
                goals += 1
            else:
                # print("Result: No Goal")
                saves += 1
                # print('\n')
        print(saves / (saves + goals))


def confidence_interval_skaters(name, n=1000000, avg=False, att = 50, confidence=0.95):
    '''
    :return:
    '''
    goalies = parse_goalies()
    goalies = pd.DataFrame(
        [goalies.loc[idx] for idx in goalies.index for _ in range(int(goalies.loc[idx]['Attempts']))]).reset_index(
        drop=True)
    g = goalies.shape[0]

    skaters = parse_skaters(n=att)
    league_mean = skaters['Goals'].sum() / skaters['Attempts'].sum()
    p = (league_mean * (1 - league_mean)) / (1 - league_mean)
    # league_median = skaters['Actual_Conversion_Rate'].median()
    # p = (league_median * (1 - league_mean)) / (1 - league_median)
    print(league_mean)
    skater = skaters.loc[name]
    if att == -1:
        att = skater['Attempts']

    distribution = []

    for i in range(n):
        goals = 0
        saves = 0
        for i in range(int(att)):
            r2 = random.randint(0, g - 1)
            # print(goalies.iloc[r2])
            # print('xG: ' + str(skaters.iloc[r1][5] / (goalies.iloc[r2][5] + skaters.iloc[r1][5])))
            if not avg:
                p1 = int(skater[5] * 100)
            else:
                p1 = int(p * 100)
            p2 = int(goalies.iloc[r2][5] * 100)
            r3 = random.randint(1, p1 + p2)
            if r3 <= p1:
                goals += 1
            else:
                saves += 1
        distribution.append(goals / (saves + goals))

    values, bins, _ = plt.hist(distribution, normed=True)
    area = sum(np.diff(bins) * values)

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

    def mean_confidence_interval(data, confidence=confidence):
        import scipy.stats as stats
        a = np.array(data)
        mean, sigma = a.mean(), a.std(ddof=1)
        conf_int_a = stats.norm.interval(confidence, loc=mean, scale=sigma)
        return conf_int_a

    ci = mean_confidence_interval(distribution)
    print(str(ci[0]) + ", " + str(ci[1]))

    plt.show()

    return distribution


def confidence_interval_goalies(name, n=1000000, avg=False, att = 50, confidence=0.95):
    '''
    :return:
    '''
    goalies = parse_skaters()
    goalies = pd.DataFrame(
        [goalies.loc[idx] for idx in goalies.index for _ in range(int(goalies.loc[idx]['Attempts']))]).reset_index(
        drop=True)
    g = goalies.shape[0]

    skaters = parse_goalies(n=att)
    league_mean = goalies['Goals'].sum() / goalies['Attempts'].sum()
    p = ((1 - league_mean) * league_mean) / (league_mean)
    # league_median = skaters['Actual_Conversion_Rate'].median()
    # p = (league_median * (1 - league_mean)) / (1 - league_median)
    # skater = skaters.loc[name]
    # if att == -1:
    #     att = skater['Attempts']

    distribution = []

    for i in range(n):
        goals = 0
        saves = 0
        for i in range(int(att)):
            r2 = random.randint(0, g - 1)
            # print(goalies.iloc[r2])
            # print('xG: ' + str(skaters.iloc[r1][5] / (goalies.iloc[r2][5] + skaters.iloc[r1][5])))
            # if not avg:
            #     p1 = int(skater[5] * 100)
            # else:
            p1 = int(p * 100)
            p2 = int(goalies.iloc[r2][5] * 100)
            r3 = random.randint(1, p1 + p2)
            if r3 <= p1:
                saves += 1
            else:
                goals += 1
        distribution.append(saves / (saves + goals))

    values, bins, _ = plt.hist(distribution, normed=True)

    def find_bin_idx_of_value(bins, value):
        """Finds the bin which the value corresponds to."""
        array = np.asarray(value)
        idx = np.digitize(array, bins)
        return idx - 1

    def mean_confidence_interval(data, confidence=confidence):
        import scipy.stats as stats
        a = np.array(data)
        mean, sigma = a.mean(), a.std(ddof=1)
        conf_int_a = stats.norm.interval(confidence, loc=mean, scale=sigma)
        return conf_int_a

    ci = mean_confidence_interval(distribution)
    print(str(ci[0]) + ", " + str(ci[1]))

    plt.show()

    return distribution


if __name__ == '__main__':
    # parse_skaters()
    # parse_goalies()
    simulate(type=2)
    # confidence_interval_goalies('David Pastrnak\pastrda01', n = 1000, avg=True, att=200, confidence=0.9)
