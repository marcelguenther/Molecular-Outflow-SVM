import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as datetime                                                   ## import datetime package
from scipy.interpolate import interp1d
from scipy.interpolate import NearestNDInterpolator
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.interpolate import CubicSpline
from scipy.interpolate import LinearNDInterpolator
from scipy.interpolate import splrep, BSpline
from scipy.interpolate import bisplrep, bisplev

def CalcStat(x, y, nbins=10):
    ## Function to bin the data and calculate the std in x and y direction
    ## Determine counts and position of each bin
    n, pos = np.histogram(x, bins=nbins)

    ## Remove empty bins
    #nbins = nbins - np.bincount(n)[0]
    no = n
    nt = np.append(n, 1)
    pos = pos[nt != 0]
    n = n[n != 0]

    ## Calculate the weighted counts to determine the mean and std
    sx, _ = np.histogram(x, bins=pos, weights=x)
    sx2, _ = np.histogram(x, bins=pos, weights=x*x)

    ## Calculte the mean and std of each bin
    xdata = sx / n
    xerr = np.sqrt(np.maximum(sx2/n - xdata*xdata, 0))

    ## Same for y
    sy, _ = np.histogram(x, bins=nbins, weights=y)
    sy2, _ = np.histogram(x, bins=nbins, weights=y*y)
    sy = sy[no != 0]
    sy2 = sy2[no != 0]

    ydata = sy / n
    yerr = np.sqrt(np.maximum(sy2/n - ydata*ydata, 0))

    ## If an error is 0 set it to None
    if np.all(xerr == 0):
        xerr = None

    if np.all(yerr == 0):
        yerr = None

    ## Return the results
    return xdata, ydata, xerr, yerr


def PlotDatabase(database, yaxis=None, xerr=None, yerr=None, split_set=False, ThesisMode=False, savename=None, savepath=None):

    #print(xerr)
    #print(xerr is None)
    #print(yerr)
    #print(yerr is None)

    ## Define color and shape list
    #colors = ["blue", "orange", "green", "red", "purple", "brown", "pink", "gray", "olive", "cyan"]
    colors = ["b", "g", "r", "c", "m", "y", "k"]
    shapes = ["P", "s", "D", "*", "o", "^", ">", "X"]

    ## Get x information
    xlabel = database.columns[0]
    xdata = database[xlabel]

    ## Number of datapoints
    ndp = xdata.nunique()
        
    ## Get number of bins

    ## Check for integer divisor between 5 and 15
    lod = []
    for i in range(5,15):
        #print(ndp, i, ndp%i)
        if ndp%i == 0:
            lod.append(i)

    ## List --> array
    lod = np.array(lod)

    ## If there is no result set the number of bins to 10 or the max number of unique datapoints
    if len(lod) == 0:
        nbins = min(10, ndp)

    ## If there is just one result take it
    elif len(lod) == 1:
        nbins = lod[0]

    ## If there are more candinates, find the closest to 10
    else:
        idx = (np.abs(lod - 10)).argmin()
        nbins = lod[idx]

    ## Initiate figure
    if ThesisMode == False:
        fig, ax = plt.subplots(1,figsize=(12, 12))
    else:
        fig, ax = plt.subplots(1,figsize=(7.47, 7.47))


    ## Split database or not
    if split_set == False:
        ## Plot all columns against the 1st
        for i in range(1,len(database.columns)):

            ## Get data, (errors) and label of column
            label = database.columns[i]
            ydata = database[label]
            if xerr is not None:
                xerri = xerr[i-1].T
            if yerr is not None:
                yerri = yerr[i-1].T

            ## If there are just a few (<20) data points, just plot them
            if len(xdata) < 20:
                ## Scatter the data points or plot errorbars
                if xerr is not None and yerr is not None:
                    ax.errorbar(xdata, ydata, xerr=xerri, yerr=yerri, fmt="%ss" %(colors[(i-1)%len(colors)]), alpha=0.5, label="%s" %(label), linewidth=3, elinewidth=.5, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=.5)
                elif xerr is not None and yerr is None:
                    ax.errorbar(xdata, ydata, xerr=xerri, fmt="%ss" %(colors[(i-1)%len(colors)]), alpha=0.5, label="%s" %(label), linewidth=3, elinewidth=.5, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=.5)
                elif xerr is None and yerr is not None:
                    #print("Prior errorbar if.")
                    ax.errorbar(xdata, ydata, yerr=yerri, fmt="%ss" %(colors[(i-1)%len(colors)]), alpha=0.5, label="%s" %(label), linewidth=3, elinewidth=1, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=2.5)
                else:
                    ax.scatter(xdata, ydata, c="%s" %(colors[(i-1)%len(colors)]), alpha=0.5, label="%s" %(label))

            else:
                ## Bin data and get the standart derivative
                xdb, ydb, xds, yds = CalcStat(xdata, ydata, nbins)

                ## Scatter the data points or plot errorbars
                if xerr is not None and yerr is not None:
                    ax.errorbar(xdata, ydata, xerr=xerri, yerr=yerri, fmt="%ss" %(colors[(i-1)%len(colors)]), alpha=0.15, label="%s - full data" %(label), linewidth=3, elinewidth=.5, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=.5)
                elif xerr is not None and yerr is None:
                    ax.errorbar(xdata, ydata, xerr=xerri, fmt="%ss" %(colors[(i-1)%len(colors)]), alpha=0.15, label="%s - full data" %(label), linewidth=3, elinewidth=.5, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=.5)
                elif xerr is None and yerr is not None:
                    #print("Prior errorbar else.")
                    ax.errorbar(xdata, ydata, yerr=yerri, fmt="%ss" %(colors[(i-1)%len(colors)]), alpha=0.15, label="%s - full data" %(label), linewidth=3, elinewidth=.5, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=.5)
                else:
                    ax.scatter(xdata, ydata, c="%s" %(colors[(i-1)%len(colors)]), alpha=0.15, label="%s - full data" %(label))

                ## Plot the binned data points
                ax.errorbar(xdb, ydb, xerr=xds, yerr=yds, fmt="%s%s" %(colors[(i-1)%len(colors)], shapes[int(np.floor((i-1)/len(colors)))]),
                            linewidth=3, elinewidth=.5, ecolor="%s" %(colors[(i-1)%len(colors)]), capsize=5, capthick=.5, label="%s - binned" %(label))

    else:
        ## Scatter the data points
        ydata = database[database.columns[1]]
        ax.scatter(xdata, ydata, c="%s" %(colors[0]), alpha=0.15, label="All data")
        
        ## Plot all columns against the 1st
        for i in range(2,len(database.columns)):

            ## Get data and label of column
            label = database.columns[i]
            cdata = database[label]

            ## Get unique class data
            ucdata = cdata.unique()

            ## Iterate over unique class data
            for ii, cd in enumerate(ucdata):

                ## Find all matching values
                cdi = cdata.index[cdata == cd].tolist()

                ## Bin data and get the standart derivative
                xdb, ydb, xds, yds = CalcStat(xdata[cdi], ydata[cdi], nbins)

                ## Plot the binned data points
                ax.errorbar(xdb, ydb, xerr=xds, yerr=yds, fmt="%s%s" %(colors[(i-1)%len(colors)], shapes[ii%len(shapes)]),
                            linewidth=3, elinewidth=.5, ecolor='k', capsize=5, capthick=.5, label="%s - %s" %(label, cd))



    ## Name axes and title
    ax.set_xlabel(r"%s" %(xlabel))


    if yaxis == "accuracy":
        
        scoremin = database[database.columns[1:]].min().min()
        scoremax = database[database.columns[1:]].max().max()
        
        pymi, pyma = ax.get_ylim()
        if pyma - pymi <= 20:
            pass
        elif pymi <= 40 and pyma >= 60:
            ax.set_ylim(-5, 105)
        elif pymi <= 40:
            ax.set_ylim(0 - abs(scoremax-pyma), pyma)
        elif pyma >= 60:
            ax.set_ylim(pymi, 100 + abs(scoremin-pymi))
        ax.set_ylabel(r"Accuracy [%]")
        ax.set_title("SVM Accuracy")

    elif yaxis == "FPI":
        ax.axhline(y=0, color="gray", linestyle="--", lw=2)
        ax.set_ylabel(r"Decrease in accuracy score")
        ax.set_title("SVM Feature Importance")

    ## Set grid and legend
    ax.grid(color='black', ls='solid', lw=0.1)
    ax.legend()

    ## Save figure if path and name are provided
    if savename != None and savepath != None:
        plt.savefig("%s%s" %(savepath, savename.replace("/", "-")), dpi='figure', format="pdf", metadata=None, bbox_inches="tight", 
                    pad_inches=0.1, backend=None)
        plt.close()

    else:
        plt.show()

## Iterpolate Database to 2d
def PlotDatabase2D(db, Set, savename=None, savepath=None):

    ## Initiate figure
    fig_int_ov_n, ax_int_ov_n = plt.subplots(len(Set), len(Set), figsize=(4*len(Set)+0.5,4*len(Set)))
    fig_int_ov_l, ax_int_ov_l = plt.subplots(len(Set), len(Set), figsize=(4*len(Set)+0.5,4*len(Set)))
    fig_int_ov_c, ax_int_ov_c = plt.subplots(len(Set), len(Set), figsize=(4*len(Set)+0.5,4*len(Set)))
    fig_int_ov_s, ax_int_ov_s = plt.subplots(len(Set), len(Set), figsize=(4*len(Set)+0.5,4*len(Set)))
    
    fig_int_pn_n, ax_int_pn_n = plt.subplots(len(Set), len(Set), figsize=(4*len(Set)+0.5,4*len(Set)))
    fig_int_pn_l, ax_int_pn_l = plt.subplots(len(Set), len(Set), figsize=(4*len(Set)+0.5,4*len(Set)))
    fig_int_pn_c, ax_int_pn_c = plt.subplots(len(Set), len(Set), figsize=(4*len(Set)+0.5,4*len(Set)))
    fig_int_pn_s, ax_int_pn_s = plt.subplots(len(Set), len(Set), figsize=(4*len(Set)+0.5,4*len(Set)))

    ## Iterate over all column combinations
    for i, columni in enumerate(Set):
        for ii, columnii in enumerate(Set):

            print("(%i|%i) of %i; time stamp: %s" %(i, ii, len(Set), datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
            
            ## Extract specific values
            ci = db[[columni]].to_numpy().reshape(-1)                       # Values of column i
            cii = db[[columnii]].to_numpy().reshape(-1)                     # Values of column ii
            sc_ov = db[["Overall score"]].to_numpy().reshape(-1)            # Overall score of point
            sc_n = db[["Outflow score"]].to_numpy().reshape(-1)             # Outflow score of point
            sc_p = db[["Non-outflow score"]].to_numpy().reshape(-1)         # Non outflow score of point
            mgi = np.linspace(ci.min(), ci.max(), 200)                      # Grid for column i
            mgii = np.linspace(cii.min(), cii.max(), 200)                   # Grid for column ii
            mgim, mgiim = np.meshgrid(mgi, mgii, indexing="ij")             # Meshgrid

            # On diagonal elements
            if i == ii:
                ## Sort values, neccessary for 1d interpol functions
                cis, scs_ov = [list(tuple) for tuple in  zip(*sorted(zip(ci, sc_ov)))]
                cis, scs_n = [list(tuple) for tuple in  zip(*sorted(zip(ci, sc_n)))]
                cis, scs_p = [list(tuple) for tuple in  zip(*sorted(zip(ci, sc_p)))]

                ## Interpolate cubic, linear, nearest neighbour, smoothing splines
                interp1d_ov_c = CubicSpline(cis, scs_ov, extrapolate=True)
                interp1d_ov_l = interp1d(cis, scs_ov, kind="linear")
                interp1d_ov_n = interp1d(cis, scs_ov, kind="nearest")
                interp1d_ov_s = splrep(cis, scs_ov, s=len(sc_ov)*100)

                interp1d_n_c = CubicSpline(cis, scs_n, extrapolate=True)
                interp1d_n_l = interp1d(cis, scs_n, kind="linear")
                interp1d_n_n = interp1d(cis, scs_n, kind="nearest")
                interp1d_n_s = splrep(cis, scs_n, s=len(sc_p)*100)

                interp1d_p_c = CubicSpline(cis, scs_p, extrapolate=True)
                interp1d_p_l = interp1d(cis, scs_p, kind="linear")
                interp1d_p_n = interp1d(cis, scs_p, kind="nearest")
                interp1d_p_s = splrep(cis, scs_p, s=len(sc_p)*100)

                ## Calculate interpolated values
                Z_ov_c = interp1d_ov_c(mgi)
                Z_ov_l = interp1d_ov_l(mgi)
                Z_ov_n = interp1d_ov_n(mgi)
                Z_ov_s = BSpline(*interp1d_ov_s)(mgi)
                
                Z_n_c = interp1d_n_c(mgi)
                Z_n_l = interp1d_n_l(mgi)
                Z_n_n = interp1d_n_n(mgi)
                Z_n_s = BSpline(*interp1d_n_s)(mgi)

                Z_p_c = interp1d_p_c(mgi)
                Z_p_l = interp1d_p_l(mgi)
                Z_p_n = interp1d_p_n(mgi)
                Z_p_s = BSpline(*interp1d_p_s)(mgi)

                ## Plot them
                ax_int_ov_l[i,ii].plot(mgi, Z_ov_l)
                ax_int_ov_c[i,ii].plot(mgi, Z_ov_c)
                ax_int_ov_n[i,ii].plot(mgi, Z_ov_n)
                ax_int_ov_s[i,ii].plot(mgi, Z_ov_s)

                ax_int_pn_l[i,ii].plot(mgi, Z_n_l, c="red", label="Outflow")
                ax_int_pn_c[i,ii].plot(mgi, Z_n_c, c="red", label="Outflow")
                ax_int_pn_n[i,ii].plot(mgi, Z_n_n, c="red", label="Outflow")
                ax_int_pn_s[i,ii].plot(mgi, Z_n_s, c="red", label="Outflow")

                ax_int_pn_l[i,ii].plot(mgi, Z_p_l, c="blue", label="Non-Outflow")
                ax_int_pn_c[i,ii].plot(mgi, Z_p_c, c="blue", label="Non-Outflow")
                ax_int_pn_n[i,ii].plot(mgi, Z_p_n, c="blue", label="Non-Outflow")
                ax_int_pn_s[i,ii].plot(mgi, Z_p_s, c="blue", label="Non-Outflow")

                ## Set plot ranges to 0 to 100%
                ax_int_ov_l[i,ii].set_ylim(0,100)
                ax_int_ov_c[i,ii].set_ylim(0,100)
                ax_int_ov_n[i,ii].set_ylim(0,100)
                ax_int_ov_s[i,ii].set_ylim(0,100)

                ax_int_pn_l[i,ii].set_ylim(0,100)
                ax_int_pn_c[i,ii].set_ylim(0,100)
                ax_int_pn_n[i,ii].set_ylim(0,100)
                ax_int_pn_s[i,ii].set_ylim(0,100)

                ## Set axis labels
                ax_int_ov_l[i,ii].set_xlabel(Set[ii])
                ax_int_ov_l[i,ii].set_ylabel("Accuracy [%]")
                ax_int_ov_c[i,ii].set_xlabel(Set[ii])
                ax_int_ov_c[i,ii].set_ylabel("Accuracy [%]")
                ax_int_ov_n[i,ii].set_xlabel(Set[ii])
                ax_int_ov_n[i,ii].set_ylabel("Accuracy [%]")
                ax_int_ov_s[i,ii].set_xlabel(Set[ii])
                ax_int_ov_s[i,ii].set_ylabel("Accuracy [%]")

                ax_int_pn_l[i,ii].set_xlabel(Set[ii])
                ax_int_pn_l[i,ii].set_ylabel("Accuracy [%]")
                ax_int_pn_c[i,ii].set_xlabel(Set[ii])
                ax_int_pn_c[i,ii].set_ylabel("Accuracy [%]")
                ax_int_pn_n[i,ii].set_xlabel(Set[ii])
                ax_int_pn_n[i,ii].set_ylabel("Accuracy [%]")
                ax_int_pn_s[i,ii].set_xlabel(Set[ii])
                ax_int_pn_s[i,ii].set_ylabel("Accuracy [%]")

                ## Add labels for the 1st outflow/non-outflow image
                if i == 0:
                    ax_int_pn_l[i,ii].legend()
                    ax_int_pn_c[i,ii].legend()
                    ax_int_pn_n[i,ii].legend()
                    ax_int_pn_s[i,ii].legend()

            ## Bottom left off diagonal elements
            elif i > ii:

                ## Interpolate
                interp2d_ov_c = CloughTocher2DInterpolator(list(zip(ci, cii)), sc_ov)
                interp2d_ov_l = LinearNDInterpolator(list(zip(ci, cii)), sc_ov)
                interp2d_ov_n = NearestNDInterpolator(list(zip(ci, cii)), sc_ov)
                interp2d_ov_s = bisplrep(ci, cii, sc_ov, s=len(ci))
                
                interp2d_n_c = CloughTocher2DInterpolator(list(zip(ci, cii)), sc_n)
                interp2d_n_l = LinearNDInterpolator(list(zip(ci, cii)), sc_n)
                interp2d_n_n = NearestNDInterpolator(list(zip(ci, cii)), sc_n)
                interp2d_n_s = bisplrep(ci, cii, sc_n, s=len(ci))

                
                ## Calculate values
                Z_ov_c = interp2d_ov_c(mgim, mgiim)
                Z_ov_l = interp2d_ov_l(mgim, mgiim)
                Z_ov_n = interp2d_ov_n(mgim, mgiim)
                Z_ov_s = bisplev(mgim[:,0], mgiim[0,:], interp2d_ov_s)

                Z_n_c = interp2d_n_c(mgim, mgiim)
                Z_n_l = interp2d_n_l(mgim, mgiim)
                Z_n_n = interp2d_n_n(mgim, mgiim)
                Z_n_s = bisplev(mgim[:,0], mgiim[0,:], interp2d_n_s)

                ## Minimal values
                fxmin = mgii[0] - (mgii[1]-mgii[0])/2
                fxmax = mgii[-1] + (mgii[1]-mgii[0])/2
                fymin = mgi[0] - (mgi[1]-mgi[0])/2
                fymax = mgi[-1] + (mgi[1]-mgi[0])/2

                ## Plot data
                img_ov_c = ax_int_ov_c[i,ii].imshow(Z_ov_c, aspect="auto", origin="lower", cmap="viridis", extent=[fxmin, fxmax, fymin, fymax], vmin=0, vmax=100)
                img_ov_l = ax_int_ov_l[i,ii].imshow(Z_ov_l, aspect="auto", origin="lower", cmap="viridis", extent=[fxmin, fxmax, fymin, fymax], vmin=0, vmax=100)
                img_ov_n = ax_int_ov_n[i,ii].imshow(Z_ov_n, aspect="auto", origin="lower", cmap="viridis", extent=[fxmin, fxmax, fymin, fymax], vmin=0, vmax=100)
                img_ov_s = ax_int_ov_s[i,ii].imshow(Z_ov_s, aspect="auto", origin="lower", cmap="viridis", extent=[fxmin, fxmax, fymin, fymax], vmin=0, vmax=100)
                
                img_n_c = ax_int_pn_c[i,ii].imshow(Z_n_c, aspect="auto", origin="lower", cmap="Reds", extent=[fxmin, fxmax, fymin, fymax], vmin=0, vmax=100)
                img_n_l = ax_int_pn_l[i,ii].imshow(Z_n_l, aspect="auto", origin="lower", cmap="Reds", extent=[fxmin, fxmax, fymin, fymax], vmin=0, vmax=100)
                img_n_n = ax_int_pn_n[i,ii].imshow(Z_n_n, aspect="auto", origin="lower", cmap="Reds", extent=[fxmin, fxmax, fymin, fymax], vmin=0, vmax=100)
                img_n_s = ax_int_pn_s[i,ii].imshow(Z_n_s, aspect="auto", origin="lower", cmap="Reds", extent=[fxmin, fxmax, fymin, fymax], vmin=0, vmax=100)

                ## Add colourbars
                im_ratio = mgim.shape[0]/mgim.data.shape[1]
                cbar_int_ov_c = plt.colorbar(img_ov_c, ax=ax_int_ov_c[i, ii], fraction=0.047*im_ratio)
                cbar_int_ov_l = plt.colorbar(img_ov_l, ax=ax_int_ov_l[i, ii], fraction=0.047*im_ratio)
                cbar_int_ov_n = plt.colorbar(img_ov_n, ax=ax_int_ov_n[i, ii], fraction=0.047*im_ratio)
                cbar_int_ov_s = plt.colorbar(img_ov_s, ax=ax_int_ov_s[i, ii], fraction=0.047*im_ratio)

                cbar_int_n_c = plt.colorbar(img_n_c, ax=ax_int_pn_c[i, ii], fraction=0.047*im_ratio)
                cbar_int_n_l = plt.colorbar(img_n_l, ax=ax_int_pn_l[i, ii], fraction=0.047*im_ratio)
                cbar_int_n_n = plt.colorbar(img_n_n, ax=ax_int_pn_n[i, ii], fraction=0.047*im_ratio)
                cbar_int_n_s = plt.colorbar(img_n_s, ax=ax_int_pn_s[i, ii], fraction=0.047*im_ratio)

                ## Set axis and colourbar labels
                ax_int_ov_l[i,ii].set_xlabel(Set[ii])
                ax_int_ov_l[i,ii].set_ylabel(Set[i])
                cbar_int_ov_c.set_label("Accuracy [%]")
                ax_int_ov_c[i,ii].set_xlabel(Set[ii])
                ax_int_ov_c[i,ii].set_ylabel(Set[i])
                cbar_int_ov_l.set_label("Accuracy [%]")
                ax_int_ov_n[i,ii].set_xlabel(Set[ii])
                ax_int_ov_n[i,ii].set_ylabel(Set[i])
                cbar_int_ov_n.set_label("Accuracy [%]")
                ax_int_ov_s[i,ii].set_xlabel(Set[ii])
                ax_int_ov_s[i,ii].set_ylabel(Set[i])
                cbar_int_ov_s.set_label("Accuracy [%]")
                ax_int_pn_l[i,ii].set_xlabel(Set[ii])
                ax_int_pn_l[i,ii].set_ylabel(Set[i])
                cbar_int_n_c.set_label("Accuracy [%]")
                ax_int_pn_c[i,ii].set_xlabel(Set[ii])
                ax_int_pn_c[i,ii].set_ylabel(Set[i])
                cbar_int_n_l.set_label("Accuracy [%]")
                ax_int_pn_n[i,ii].set_xlabel(Set[ii])
                ax_int_pn_n[i,ii].set_ylabel(Set[i])
                cbar_int_n_n.set_label("Accuracy [%]")
                ax_int_pn_s[i,ii].set_xlabel(Set[ii])
                ax_int_pn_s[i,ii].set_ylabel(Set[i])
                cbar_int_n_s.set_label("Accuracy [%]")
    
            ## Top right off diagonal elements
            elif i < ii:

                ## Interpolate                
                interp2d_p_c = CloughTocher2DInterpolator(list(zip(ci, cii)), sc_p)
                interp2d_p_l = LinearNDInterpolator(list(zip(ci, cii)), sc_p)
                interp2d_p_n = NearestNDInterpolator(list(zip(ci, cii)), sc_p)
                interp2d_p_s = bisplrep(ci, cii, sc_p, s=len(ci))

                
                ## Calculate values
                Z_p_c = interp2d_p_c(mgim, mgiim)
                Z_p_l = interp2d_p_l(mgim, mgiim)
                Z_p_n = interp2d_p_n(mgim, mgiim)
                Z_p_s = bisplev(mgim[:,0], mgiim[0,:], interp2d_p_s)

                ## Minimal values
                fxmin = mgii[0] - (mgii[1]-mgii[0])/2
                fxmax = mgii[-1] + (mgii[1]-mgii[0])/2
                fymin = mgi[0] - (mgi[1]-mgi[0])/2
                fymax = mgi[-1] + (mgi[1]-mgi[0])/2

                ## Plot data
                img_p_c = ax_int_pn_c[i,ii].imshow(Z_p_c, aspect="auto", origin="lower", cmap="Blues", extent=[fxmin, fxmax, fymin, fymax], vmin=0, vmax=100)
                img_p_l = ax_int_pn_l[i,ii].imshow(Z_p_l, aspect="auto", origin="lower", cmap="Blues", extent=[fxmin, fxmax, fymin, fymax], vmin=0, vmax=100)
                img_p_n = ax_int_pn_n[i,ii].imshow(Z_p_n, aspect="auto", origin="lower", cmap="Blues", extent=[fxmin, fxmax, fymin, fymax], vmin=0, vmax=100)
                img_p_s = ax_int_pn_s[i,ii].imshow(Z_p_s, aspect="auto", origin="lower", cmap="Blues", extent=[fxmin, fxmax, fymin, fymax], vmin=0, vmax=100)

                ## Add colourbars
                im_ratio = mgim.shape[0]/mgim.data.shape[1]
                cbar_int_p_c = plt.colorbar(img_p_c, ax=ax_int_pn_c[i, ii], fraction=0.047*im_ratio)
                cbar_int_p_l = plt.colorbar(img_p_l, ax=ax_int_pn_l[i, ii], fraction=0.047*im_ratio)
                cbar_int_p_n = plt.colorbar(img_p_n, ax=ax_int_pn_n[i, ii], fraction=0.047*im_ratio)
                cbar_int_p_s = plt.colorbar(img_p_s, ax=ax_int_pn_s[i, ii], fraction=0.047*im_ratio)

                ## Set axis and colourbar labels
                ax_int_pn_l[i,ii].set_xlabel(Set[ii])
                ax_int_pn_l[i,ii].set_ylabel(Set[i])
                cbar_int_p_c.set_label("Accuracy [%]")
                ax_int_pn_c[i,ii].set_xlabel(Set[ii])
                ax_int_pn_c[i,ii].set_ylabel(Set[i])
                cbar_int_p_l.set_label("Accuracy [%]")
                ax_int_pn_n[i,ii].set_xlabel(Set[ii])
                ax_int_pn_n[i,ii].set_ylabel(Set[i])
                cbar_int_p_n.set_label("Accuracy [%]")
                ax_int_pn_s[i,ii].set_xlabel(Set[ii])
                ax_int_pn_s[i,ii].set_ylabel(Set[i])
                cbar_int_p_s.set_label("Accuracy [%]")

    ## Set titles
    fig_int_ov_c.suptitle("Cubic interpolation")
    fig_int_ov_l.suptitle("Linear interpolation")
    fig_int_ov_n.suptitle("NN interpolation")
    fig_int_ov_s.suptitle("Smoothing Splines interpolation")
    fig_int_pn_c.suptitle("Cubic interpolation")
    fig_int_pn_l.suptitle("Linear interpolation")
    fig_int_pn_n.suptitle("NN interpolation")
    fig_int_pn_s.suptitle("Smoothing Splines interpolation")


    ## Remove empty plots
    for i in range(len(Set)):
        for ii in range(len(Set)):
            if i < ii:
                fig_int_ov_c.delaxes(ax_int_ov_c[i, ii])
                fig_int_ov_l.delaxes(ax_int_ov_l[i, ii])
                fig_int_ov_n.delaxes(ax_int_ov_n[i, ii])
                fig_int_ov_s.delaxes(ax_int_ov_s[i, ii])

    ## Save figure if path and name are provided
    if savepath != None:
        time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        fig_int_ov_c.savefig("%sPerformance_Overall_Overview_Cubic_%s.pdf" %(savepath, time_stamp), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)
        fig_int_ov_l.savefig("%sPerformance_Overall_Overview_Linear_%s.pdf" %(savepath, time_stamp), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        fig_int_ov_n.savefig("%sPerformance_Overall_Overview_NN_%s.pdf" %(savepath, time_stamp), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        fig_int_ov_s.savefig("%sPerformance_Overall_Overview_SS_%s.pdf" %(savepath, time_stamp), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        fig_int_pn_c.savefig("%sPerformance_OfNoOf_Overview_Cubic_%s.pdf" %(savepath, time_stamp), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        fig_int_pn_l.savefig("%sPerformance_OfNoOf_Overview_Linear_%s.pdf" %(savepath, time_stamp), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        fig_int_pn_n.savefig("%sPerformance_OfNoOf_Overview_NN_%s.pdf" %(savepath, time_stamp), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        fig_int_pn_s.savefig("%sPerformance_OfNoOf_Overview_SS_%s.pdf" %(savepath, time_stamp), dpi='figure', format="pdf",
                    metadata=None, bbox_inches="tight", pad_inches=0.1, backend=None)

        plt.close("all")

    else:
        plt.show()    

def PlotResults(CalcParameters, mode):

    ## Read in data frames
    if CalcParameters["DatabaseAutomatic"] == True or CalcParameters["DatabaseManual"] == True:
        ## Read the score data frame
        if mode == "full":
            data_pd = pd.read_csv("%s%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]), index_col=0)

        elif mode == "test":
            data_pd = pd.read_csv("%sTest-Mode_%s" %(CalcParameters["DatePath"], CalcParameters["database_short"]), index_col=0)

    
        ## Read the feature importance data frame 
        if CalcParameters["CHECK_FI"] and "Mask name" in data_pd.columns and CalcParameters["MASK"] == True:
            data_pd_fi = pd.read_csv("%sFeatureImportance.csv" %(CalcParameters["DatePath"]))

            #print(data_pd_fi)
            #print()

            whiskers_data = data_pd_fi.drop([col for col in data_pd_fi.columns if "whiskers length" not in col], axis=1).to_numpy()
            #print(whiskers_data)
            whsikers_length = np.zeros(shape=(int(len(whiskers_data[0])/2),len(whiskers_data),2))
            for i in range(int(len(whiskers_data[0])/2)):
                whsikers_length[i] = whiskers_data[:,[2*i,2*i+1]]
            #print(whsikers_length)
            #print()
            #print()
            
            data_pd_fi = data_pd_fi.drop([col for col in data_pd_fi.columns if "whiskers length" in col], axis=1)
            #print(data_pd_fi)
            #print()

            feature_cols = [col for col in data_pd_fi.columns if "Feature" in col]
            #print(feature_cols)
            #print()


    ## Analyze automatic set (and feature importance set)
    if CalcParameters['DatabaseAutomatic'] == True:

        Automatic_Set = CalcParameters['AutomaticSet'] 

        ## Iterate over sets
        for column in Automatic_Set:

            ## Get sub data set
            dataset = data_pd[[column, "Overall score", "Balanced score", "Outflow score", "Non-outflow score"]]
            
            ## Get a time stamp to ensure the plot names are unique
            time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            ## Generate the plots
            if mode == "full":
                PlotDatabase(dataset, yaxis="accuracy", savename="%s-Automatic-%s.pdf" %(column, time_stamp), ThesisMode=CalcParameters["ThesisMode"], savepath=CalcParameters["PlotPath"]) 
            elif mode == "test":
                PlotDatabase(dataset, yaxis="accuracy", savename="Test-Mode_%s-Automatic-%s.pdf" %(column, time_stamp), ThesisMode=CalcParameters["ThesisMode"], savepath=CalcParameters["PlotPath"]) 


            if CalcParameters["CHECK_FI"] and "Mask name" in data_pd.columns and CalcParameters["MASK"] == True:

                ## Get sub data set
                dataset = data_pd_fi[np.append(column, feature_cols)]

                ## Generate the plots
                if mode == "full":
                    PlotDatabase(dataset, yaxis="FPI", yerr=whsikers_length, ThesisMode=CalcParameters["ThesisMode"], savename="%s-Automatic_FPI-%s.pdf" %(column, time_stamp), savepath=CalcParameters["PlotPath"]) 
                elif mode == "test":
                    PlotDatabase(dataset, yaxis="FPI", yerr=whsikers_length, ThesisMode=CalcParameters["ThesisMode"], savename="Test-Mode_%s-Automatic_FPI-%s.pdf" %(column, time_stamp), savepath=CalcParameters["PlotPath"]) 
        
        ## Interpolate scores to 1d/2d with all combinations
        if len(Automatic_Set) >= 2:
            PlotDatabase2D(data_pd, Automatic_Set)

    ## Analyze manual set
    if CalcParameters['DatabaseManual'] == True:

        Manual_Set_Main = CalcParameters['ManualSetMain']
        Manual_Set_Sub = CalcParameters['ManualSetSub']

        ## Iterate over sets
        for Main_Set, Sub_Set in zip(Manual_Set_Main, Manual_Set_Sub):

            ## Create sub data sets to plot them
            Main_Column = data_pd[Main_Set]
            Score_Column = data_pd["Overall score"]
            Sub_Column = data_pd[Sub_Set]

            dataset = pd.concat([Main_Column, Score_Column, Sub_Column], axis=1)
            
            ## Get a time stamp to ensure the plot names are unique
            time_stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            ## Generate the plots
            if mode == "full":
                PlotDatabase(dataset, ThesisMode=CalcParameters["ThesisMode"], split_set = True, savename="%s-Manual-%s.pdf" %(column, time_stamp), savepath=CalcParameters["PlotPath"])
                
            elif mode == "test":
                PlotDatabase(dataset, ThesisMode=CalcParameters["ThesisMode"], split_set = True, savename="Test-Mode_%s-Manual-%s.pdf" %(column, time_stamp), savepath=CalcParameters["PlotPath"])

    return

if __name__ == "__main__":
    #print("Hello Word!")

    #MyParameters = {}

    #MyParameters["DatabaseAutomatic"] = True
    #MyParameters["DatabaseManual"] = False
    #MyParameters["DatePath"] = "D:/Phillip/Studium/Masterarbeit/DMOSVM/Data/MultiRun_2023-05-30_14-22-48/Train-Test-Set_2023-05-30_14-22-48-Run_0/SVMMasterTestInclination.xml_2023-05-30_14-22-48_Test_Angle_Database/"
    #MyParameters["database_short"] = "Database_short.csv"
    #MyParameters["MASK"] = True
    #MyParameters["AutomaticSet"] = ["Theta [deg]"]
    #MyParameters["CHECK_FI"] = True
    #MyParameters["ThesisMode"] = False
    ##MyParameters["PlotPath"] = "C:/Users/Phillip/Studium/Master/Sommersemester21/Masterarbeit/Thesis/Pictures/"
    #MyParameters["PlotPath"] = None


    #PlotResults(MyParameters, "full")
    #print("Hello Word!")

    #MyParameters = {}

    #MyParameters["DatabaseAutomatic"] = True
    #MyParameters["DatabaseManual"] = False
    #MyParameters["DatePath"] = "D:/Phillip/Studium/Masterarbeit/DMOSVM/Data/MultiRun_2023-05-24_12-28-04/Train-Test-Set_2023-05-24_12-28-05-Run_0/SVMMasterTrain.xml_2023-05-24_12-28-05_Train_Light_Low_Noise_Database/"
    #MyParameters["database_short"] = "Database_short.csv"
    #MyParameters["MASK"] = True
    #MyParameters["AutomaticSet"] = ["Theta [deg]"]
    #MyParameters["CHECK_FI"] = True
    #MyParameters["ThesisMode"] = False
    #MyParameters["PlotPath"] = "C:/Users/Phillip/Studium/Master/Sommersemester21/Masterarbeit/Thesis/Pictures/"
    ##MyParameters["PlotPath"] = None


    #PlotResults(MyParameters, "full")
    ##print("Hello Word!")

    MyParameters = {}

    MyParameters["DatabaseAutomatic"] = True
    MyParameters["DatabaseManual"] = False
    MyParameters["DatePath"] = "D:/Phillip/Studium/Masterarbeit/DMOSVM/Data/MultiRun_2023-05-24_12-28-04/Train-Test-Set_2023-05-24_12-28-05-Run_0/SVMMasterTestMedium.xml_2023-05-24_15-51-01_Train_Light_Medium_Noise_Database/"
    MyParameters["database_short"] = "Database_short.csv"
    MyParameters["MASK"] = True
    MyParameters["AutomaticSet"] = ["Theta [deg]"]
    MyParameters["CHECK_FI"] = True
    MyParameters["ThesisMode"] = False
    #MyParameters["PlotPath"] = "C:/Users/Phillip/Studium/Master/Sommersemester21/Masterarbeit/Thesis/Pictures/"
    MyParameters["PlotPath"] = None


    PlotResults(MyParameters, "full")
    #print("Hello Word!")

    MyParameters = {}

    MyParameters["DatabaseAutomatic"] = True
    MyParameters["DatabaseManual"] = False
    MyParameters["DatePath"] = "D:/Phillip/Studium/Masterarbeit/DMOSVM/Data/MultiRun_2023-05-24_12-28-04/Train-Test-Set_2023-05-24_12-28-05-Run_0/SVMMasterTestHigh.xml_2023-05-24_19-31-38_Train_Light_High_Noise_Database/"
    MyParameters["database_short"] = "Database_short.csv"
    MyParameters["MASK"] = True
    MyParameters["AutomaticSet"] = ["Theta [deg]"]
    MyParameters["CHECK_FI"] = True
    MyParameters["ThesisMode"] = False
    #MyParameters["PlotPath"] = "C:/Users/Phillip/Studium/Master/Sommersemester21/Masterarbeit/Thesis/Pictures/"
    MyParameters["PlotPath"] = None


    PlotResults(MyParameters, "full")

    #MyParameters = {}

    #MyParameters["DatabaseAutomatic"] = True
    #MyParameters["DatabaseManual"] = False
    #MyParameters["DatePath"] = "D:/Phillip/Studium/Masterarbeit/DMOSVM/Data/MultiRun_2023-05-30_14-22-48/Train-Test-Set_2023-05-30_14-22-48-Run_0/SVMMasterTestInclination.xml_2023-05-30_14-22-48_Test_Angle_Database/"
    #MyParameters["database_short"] = "Database_short.csv"
    #MyParameters["MASK"] = True
    #MyParameters["AutomaticSet"] = ["Theta [deg]"]
    #MyParameters["CHECK_FI"] = True
    #MyParameters["ThesisMode"] = False
    #MyParameters["PlotPath"] = "C:/Users/Phillip/Studium/Master/Sommersemester21/Masterarbeit/Thesis/Pictures/"
    ##MyParameters["PlotPath"] = None


    #PlotResults(MyParameters, "full")
'''
if __name__ == "__main__":
    np.random.RandomState(0)
    ga = 0.1
    gb = 0.95
    gr = 0.025

    ba = 0.5
    bb = 0.8
    br = 0.05

    na = 3

    angs = np.arccos(np.linspace(start=0,stop=1,num=11)) * 180/np.pi
    anga = np.array(na*[angs]).T.reshape(na*len(angs))

    print(angs)

    idei = np.ones_like(anga)
    gooi = ga * np.cos(anga*np.pi/180) + (gb-ga)
    badi = ba * np.cos(anga*np.pi/180) + (bb-ba)

    idea = idei
    gooa = gooi + np.random.normal(loc=0, scale=gr, size=len(anga))
    bada = badi + np.random.normal(loc=0, scale=br, size=len(anga))

    gooa = np.minimum(gooa, 0.99)
    bada = np.minimum(bada, 0.99)

    idea *= 100
    gooa *= 100
    bada *= 100

    claa1 = np.random.randint(0,2,np.shape(anga))
    claa2 = np.random.randint(0,3,np.shape(anga))
    mcla1 = gooa - claa1*20
    mcla2 = gooa - claa1*20 - claa2*30
    #mcla1 = idea - claa1*20
    #mcla2 = idea - claa1*20 - claa2*40


    data=np.array([anga, mcla2]).T
    database = pd.DataFrame(data=data, columns=["Inclination", "Model"])
    print(database.head())
    PlotDatabase(database)

    #data=np.array([anga, idea, gooa, bada]).T
    #database = pd.DataFrame(data=data, columns=["Inclination", "Ideal model", "Good model", "Bad model"])
    #PlotDatabase(database) 

    #data=np.array([anga, gooa]).T
    #database = pd.DataFrame(data=data, columns=["Inclination", "Good model"])
    #PlotDatabase(database)

    #data=np.array([anga, mcla2]).T
    #database = pd.DataFrame(data=data, columns=["Inclination", "Model"])
    #print(database.head)
    #PlotDatabase(database)

    #data=np.array([anga, mcla1, claa1]).T
    #database = pd.DataFrame(data=data, columns=["Inclination", "Good model", "Class 1"])
    #print(database.head)
    #print(database["Class"].unique())
    #PlotDatabase(database, split_set=True)

    #data=np.array([anga, mcla2, claa1, claa2]).T
    #database = pd.DataFrame(data=data, columns=["Inclination", "Good model", "Class 1", "Class 2"])
    #print(database.head)
    #print(database["Class"].unique())
    #PlotDatabase(database, split_set=True)
'''