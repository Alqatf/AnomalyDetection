import datetime
import math

import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib import gridspec

from data import model
import preprocessing

def generate_plots(key, dataset, log=False):
    df = data.load_dataframe(dataset)
    dates, values = data.pick_headers(df, dataset["header"], log)
    
    fig, ax = plt.subplots()
    ax.plot(dates, values, '-')
    
    # format the ticks
    ax.autoscale_view()
    
    # format the coords message box
    def price(x): return '%1.2f'%x
    ax.fmt_xdata = DateFormatter('%Y')
    ax.fmt_ydata = price
    ax.grid(True)
    plt.xlim(dataset["from"],dataset["to"])
    fig.autofmt_xdate()

    plt.title(dataset['title']) 
    plt.xlabel(dataset['xlabel'])

    if True == log:
        plt.ylabel(dataset['ylabel'] + " logscale")
        fig.savefig("./plots/plot_"+key+"_log.png")
    else :
        plt.ylabel(dataset['ylabel'])
        fig.savefig("./plots/plot_"+key+".png")
    plt.close()

if __name__ == '__main__':
    import sys
    if sys.version_info < (2, 7) or sys.version_info >= (3, 0):
        print ("Requires Python 2.7.x")
        exit()
    del sys

    headerfile = './data/kddcup.names'
    datafile = './data/KDDTrain+_20Percent.txt'

    headers, attacks = model.load_headers(headerfile)
    df = model.load_dataframe(datafile,headers,datasize=100)

    protocols = list(set(df['protocol_type']))
    services = list(set(df['service']))
    flags = list(set(df['flag']))

    df = preprocessing.discretize_elems_in_list(df,attacks,'attack')
    df = preprocessing.discretize_elems_in_list(df,protocols,'protocol_type')
    df = preprocessing.discretize_elems_in_list(df,services,'service')
    df = preprocessing.discretize_elems_in_list(df,flags,'flag')

    scaled=['duration','src_bytes','dst_bytes','num_root','num_compromised','num_file_creations','count','srv_count','dst_host_count', 'dst_host_srv_count']
    df = preprocessing.scale_bignums_to_log2(df,scaled)

    for key in headers:
        fig, ax = plt.subplots()
        ax.autoscale_view()
        ax.grid(True)

        histdist = plt.hist(df[key], 5, normed=True, alpha=0.2)
        plt.ylabel(key + " (logscale)")
        fig.savefig("./plots/plot_"+key+"_log.png")

        plt.close()

