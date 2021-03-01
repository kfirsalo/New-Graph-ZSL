import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import chi2
import random


def log_likelihood_single_sample(x_i, alpha_p, xm=5):
    likelihood = alpha_p * (xm ** alpha_p) / (x_i ** (alpha_p + 1))
    log_likelihood = np.log(likelihood)
    return log_likelihood


def log_likelihood_all_samples(samples, alpha_p, xm=5):
    log_likelihood_vec = []
    for i in range(samples.size):
        x_i = samples[i]
        log_likelihood_vec.append(log_likelihood_single_sample(x_i, alpha_p, xm))
    log_likelihood_vec = np.asarray(log_likelihood_vec)
    final_sum = np.sum(log_likelihood_vec)
    return final_sum


def calculate_confidence_intervals(total, num_of_samples, c_i, xm, alpha):
    L_list = np.zeros(shape=(1, total))
    U_list = np.zeros(shape=(1, total))
    mean_list = np.zeros(shape=(1, total))

    for i in range(total):
        samples = (np.random.pareto(alpha, num_of_samples) + 1) * xm  # vector
        mean_samples = np.mean(samples)
        std_samples = np.std(samples)
        E_L_bound = mean_samples - c_i * std_samples / math.sqrt(num_of_samples)
        E_U_bound = mean_samples + c_i * std_samples / math.sqrt(num_of_samples)
        alpha_L_bound = E_U_bound / (E_U_bound - 5)
        alpha_U_bound = E_L_bound / (E_L_bound - 5)
        L_list[0, i] = alpha_L_bound
        U_list[0, i] = alpha_U_bound
        mean_list[0, i] = mean_samples

    return L_list, U_list, mean_list


"""
4a
"""
x_m, alpha = 5, 7.
# drawing samples from distribution
samples = (np.random.pareto(alpha, 1000) + 1) * x_m
count, bins, _ = plt.hist(samples, 100, density=True)
fit = alpha*x_m**alpha / bins**(alpha+1)
plt.plot(bins, max(count)*fit/max(fit), linewidth=2, color='r')
plt.xlabel('bins', fontsize=15)
plt.ylabel('probability density', fontsize=15)
plt.title('Probability Density Function', fontsize=15)
plt.grid(b=True, color='grey', alpha=0.3, linestyle='-.', linewidth=2)
plt.rcParams['figure.figsize'] = [8, 8]
plt.show()

"""
4b
"""
x_vec = list(np.arange(5, 8.5, 0.01))
y_vec = []
for i in range(len(x_vec)):
    alpha_p = x_vec[i]
    final_sum = log_likelihood_all_samples(samples, alpha_p, x_m)
    y_vec.append(final_sum)
plt.figure()
plt.plot(x_vec, y_vec)
plt.xlabel('alpha', fontsize=15)
plt.ylabel('Log Likelihood', fontsize=15)
plt.show()

"""
4c
"""
max_value = max(y_vec)
argmax_value = x_vec[np.argmax(y_vec)]
print(max_value, argmax_value)


"""
4d
"""


total = 100
alphaP = 0.05
num_of_samples = 1000
IL = np.zeros(shape=(1, total))
IU = np.zeros(shape=(1, total))
for rel in range(total):
    x = (np.random.pareto(alpha, 1000) + 1) * x_m
    a = chi2.ppf(alphaP/2, df=2*num_of_samples)
    IL[0, rel] = a/(2*sum(np.log(x/x_m)))
    b = chi2.ppf(1-(alphaP/2), df=2*num_of_samples)
    IU[0, rel] = b/(2*sum(np.log(x/x_m)))

y = (IU[0]+IL[0])/2
x = np.arange(0, 100)
yerror = (IU[0] - IL[0]).T/2
plt.figure(figsize=(20, 10))
plt.errorbar(x, y, yerr=yerror, fmt='.b')
COUNT = 0
for i, val in enumerate(y):
    if 7 < IL[0, i] or 7 > IU[0, i]:
        plt.errorbar(i, y[i], yerr=yerror[i], fmt='.r')


plt.plot([0, 100], [7, 7], 'k-', lw=2, color='green')
plt.show()


"""
4e
"""

N = 10000
alpha = 1.1
K = 100
x_m = 5
sizes = [1, 100, 10000, 100000, 100000000]
# sizes = [1, 100, 1000]
def func3():

    graph = []
    graph_hist = []
    colors = ['red', 'b', 'g', 'orange', 'purple']

    for j, size in enumerate(sizes):

        print("N = ", size)
        values = []

        for k in range(K):

            print(k)
            sum = 0
            for i in range(size):
                # x = (np.random.pareto(alpha, size) + 1) * x_m
                u = random.random()
                p = math.pow(1 - u, 1 / alpha)
                x = x_m / p
                sum += x

            sum /= size
            values.append(sum)

        bins = [0 + x for x in range(201)]
        hist, bin_edges = np.histogram(np.array(values), bins)
        count, bins, _ = plt.hist(hist, bins=a, color=colors[j], density=True)
        fit = alpha * x_m ** alpha / bins ** (alpha + 1)
        plt.plot(bins, max(count) * fit / max(fit), linewidth=2, color='r')
        graph_hist.append(hist)
        graph.append(values)
    plt.show()


func3()