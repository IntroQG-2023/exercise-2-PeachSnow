"""Functions used in the Introduction to Quantitative Geology course"""

# Import any modules needed in your functions here
import math

# Define your new functions below
def mean(numbers):
    return sum(numbers) / len(numbers)

def stddev(numbers):
    mu = mean(numbers)
    variance = sum((x - mu) ** 2 for x in numbers) / len(numbers)
    return math.sqrt(variance)

from math import sqrt
def stderr(data):
    n = len(data)
    if n <= 1:
        raise ValueError("Standard error requires at least two data points")
    std_dev = stddev(data) 
    return std_dev / sqrt(n)

import numpy as np
def gaussian(mean, stddev, x_values):
    x_values = np.array(x_values)  
    constant = 1 / (stddev * np.sqrt(2 * np.pi))
    exponent = np.exp(-((x_values - mean) ** 2) / (2 * stddev ** 2))
    return constant * exponent

def linregress(x, y):
    """
    Calculate the slope (B) and y-intercept (A) for an unweighted linear regression on x and y data.
    """
    N = len(x) 
    sum_x = sum(x) #x -- list or array of x values
    sum_y = sum(y) #y -- list or array of y values
    sum_x2 = sum(xi**2 for xi in x)
    sum_xy = sum(xi*yi for xi, yi in zip(x, y))
    # Calculate Delta
    Delta = N * sum_x2 - sum_x**2
    # Calculate A and B
    A = (sum_x2 * sum_y - sum_x * sum_xy) / Delta
    B = (N * sum_xy - sum_x * sum_y) / Delta #A, B -- y-intercept and slope of the regression line
    return A, B

def pearson(x, y):
    """
    Calculate the Pearson correlation coefficient r for two variables x and y.
    """
    # Calculate the mean of the x and y values
    mean_x = mean(x) # x: list or array, x-values
    mean_y = mean(y) # y: list or array, y-values
    # Calculate the differences from the means
    diff_x = [xi - mean_x for xi in x]
    diff_y = [yi - mean_y for yi in y]
    # Calculate the numerator of the correlation coefficient
    numerator = sum(dxi * dyi for dxi, dyi in zip(diff_x, diff_y))
    # Calculate the denominator of the correlation coefficient
    denominator = sqrt(sum(dxi**2 for dxi in diff_x) * sum(dyi**2 for dyi in diff_y))
    # Calculate the correlation coefficient
    r = numerator / denominator # r: float, Pearson correlation coefficient
    return r

def chi_squared(observed, expected, std_dev):
    """
    Calculate the reduced chi-squared goodness-of-fit measure.
    """
    observed = np.array(observed) # array-like, observed data values
    expected = np.array(expected) # array-like, expected data values based on the model
    std_dev = np.array(std_dev)   # array-like, standard deviation of the observed data
    # Calculate the number of observations based on the length of the observed data
    N = len(observed)
    # Calculate the chi-squared sum
    chi2_sum = np.sum(((observed - expected) ** 2) / std_dev ** 2)
    # Calculate the reduced chi-squared value
    chi2 = chi2_sum / N  # chi2 -- float, the reduced chi-squared value
    return chi2




