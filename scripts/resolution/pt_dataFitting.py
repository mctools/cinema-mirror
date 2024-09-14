#!/usr/bin/env python3

import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
import matplotlib.pyplot as plt

def gaussian(x, a, x0, sigma): 
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) 

def gaussian_fit(x, y):
    popt, pcov =  curve_fit(gaussian, x, y)
    mean = popt[1]
    std = popt[2]
    return mean, std

def plot_bar(x, y, normalize=False, yscale=1):
    if normalize:
        y = y / y.sum()
    else:
        pass
    y = y * yscale
    xleftEdge = x[:-1]
    plt.bar(xleftEdge, y, width=np.diff(x)*1, align='edge', label="Histogram data")


def gaussian_plot(x, y, bounds, scale=1):
    xleftEdge = x[:-1]
    popt, pcov = curve_fit(gaussian, xleftEdge, y, bounds=bounds)
    print(popt[0], popt[1], popt[2])
    yfit = gaussian(x, popt[0], popt[1], popt[2])
    plt.bar(xleftEdge, y, width=np.diff(x), align='edge', label="Histogram data")
    plt.plot(x, yfit, c='r', label=f"Gaussian fitting curve")
    plt.title(f"Gaussian fitting mean={popt[1]*scale:.6f}\n \
              ste={popt[2]*scale:.6f}, scale={scale:.1e}")
    
class HistPost():
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y

    def getAccWeight(self):
        return self.y.sum()
    
    def getCentre(self):
        return self.x[:-1] + 0.5 * np.diff(self.x)
    
    def getWeight(self):
        return self.y
    
    def getMean(self):
        if self.getAccWeight()==0:
            return 0
        else:
            return (self.getCentre() * self.getWeight()).sum()/self.getAccWeight()


    def getVar(self):
        """
        Get Variance
        """
        if self.getAccWeight()==0:
            return 0
        else:
            mean = self.getMean()
            var = ((self.getCentre() - mean)**2 * self.getWeight()).sum() / self.getAccWeight()
        return var 
    
    def getStd(self):
        """
        Get Standard deviation
        """
        return np.sqrt(self.getVar())