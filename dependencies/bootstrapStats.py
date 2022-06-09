import numpy as np
from random import choices
import pandas as pd

from numpy.lib.function_base import percentile

def bootstrap(sample, statistic=np.mean, n_resamples=1000):
    sample_stat = statistic(sample)
    statistics = []
    for _ in range(n_resamples):
        resample = choices(sample, k=len(sample))
        stat = statistic(resample)
        statistics.append(stat - sample_stat)

    statistics.sort()
    
    return statistics, sample_stat

def bootstrap_conf(sample, statistic=np.mean, n_resamples=1000, alpha=0.95, diff_or_value=1):
    statistics, sample_stat = bootstrap(sample, statistic, n_resamples)
    
    upper = percentile(statistics, 100 * (1-alpha)/2)
    lower = percentile(statistics, 100 * alpha+((1-alpha)/2))

    #print(min(statistics), max(statistics))

    if diff_or_value:
        return lower, abs(upper)
    else:
        return abs(sample_stat - lower), abs(upper - sample_stat)

def bootstrap_prob(sample, value, statistic=np.mean, n_resamples=1000):
    statistics, _ = bootstrap(sample, statistic, n_resamples)

    return sum(np.array(statistics) > value) / len(statistics)


if __name__ == "__main__":
    mu, sigma = 0.5, 1.0 # mean and standard deviation

    #sample = np.random.normal(mu, sigma, 100)
    data = pd.read_csv("of_data.txt")
    duration = data["eruptions"].values
    sample = duration * 60.0
    print(f"Bootstrap median 90% conf bounds: {bootstrap_conf(sample, statistic=np.median, n_resamples=1000, alpha=0.90, diff_or_value=0)}")
    print(np.median(sample))
    print(f"Probability of |avg(x) - \mu| > 5: {bootstrap_prob(sample, 5, statistic=np.mean, n_resamples=10000) + (1 - bootstrap_prob(sample, -5, statistic=np.mean, n_resamples=10000))}")

    