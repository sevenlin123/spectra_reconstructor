import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import glob
import sys
import os

'''
This file contains the NIRCam filter information and functions to plot them. 
Data downloaded from https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters#gsc.tab=0
'''

class NIRCamFilters:
    def __init__(self):
        self.wfilters = glob.glob('../data/throughput_curves/F???W_*txt')
        self.mfilters = glob.glob('../data/throughput_curves/F???M_*txt')
        self.uwfilters = glob.glob('../data/throughput_curves/F???W2_*txt')
        self.wfilters.sort()
        self.mfilters.sort()
        self.uwfilters.sort()
        self.f_bandwidths = {}
        self.f_bandcenters = {}
        self.transmissions = {}

        all_filters_set = [self.wfilters, self.mfilters, self.uwfilters]
        for fset in all_filters_set:
            for f in fset:
                fname = f.split('/')[-1].split('_')[0]
                trans = pd.read_csv(f, sep="\s+", skiprows=1, names=['wl', 'thro'])
                if 'W2' in fname:
                    center = float(fname.strip('F')[:-2])/100
                else:
                    center = float(fname.strip('F')[:-1])/100
    
                width = trans.wl[trans.thro>0.1].max() - trans.wl[trans.thro>0.1].min()
                self.transmissions[fname] = trans
                self.f_bandwidths[fname] = (trans.wl[trans.thro>0.1].max(), trans.wl[trans.thro>0.1].min())
                self.f_bandcenters[fname] = center

        self.fnames = self.f_bandwidths.keys()    

    def plot_filters(self, fname):
        trans = self.transmissions[fname]
        width = self.f_bandwidths[fname][0] - self.f_bandwidths[fname][1]
        plt.plot(trans.wl, trans.thro, label=f'{fname}: {width:.2f}um')

    def plot_filters_set(self, fset): # fset =  a list of fnames
        for fname in fset:  
            self.plot_filters(fname)    

    def bandwidth(self, fname):
        width = self.f_bandwidths[fname][0] - self.f_bandwidths[fname][1]
        return width
    
    def bandcenter(self, fname):
        return self.f_bandcenters[fname]

    def bandwidth_set(self, fset):
        widths = []
        for fname in fset:
            widths.append(self.bandwidth(fname))
        return widths
    
    def bandcenters_set(self, fset):
        centers = []
        for fname in fset:
            centers.append(self.bandcenter(fname))
        return centers



if __name__ == "__main__":
    filter_obj = NIRCamFilters()
    filter_obj.plot_filters_set(['F070W', 'F090W', 'F115W', 'F150W', 'F200W'])
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Throughput')
    plt.legend()
    plt.show()