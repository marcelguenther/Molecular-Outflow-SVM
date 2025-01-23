Describtion:
This is a python tool to identify objects (e.g. molecular outflows) in fits files. It goes
through all velocity channels and marks all values above a certain sigma level. Connected
pixels will be grouped. Each group will be labeled and each can be selected and deselected
with a command in the console. Additionally, it is possible to select all pixels above the
sigma level in a certain region. Right now, it is possible to create rectangular and polynimal
regions. Furthermore, all pixles of the slice can be deselected at once.
After the object is colmpletly identified, the tool creates a matching mask which is going
to be saved inside a pickled file. At the end of this file is a code to read the mask from
the file out.
"""

##############################################################################
### Include packages

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.cm as mcm
from astropy.io import fits
from scipy import ndimage
from PIL import Image, ImageDraw


def create_outflow_mask(OUTFLOW_NAME, FILE_CUBE):
##############################################################################
    ### Reading data
    data_cube = fits.getdata(FILE_CUBE, ext=0)\
        if fits.getheader(FILE_CUBE)['NAXIS'] == 3\
        else fits.getdata(FILE_CUBE, ext=0)[0]
    
    ## Creating a array to buffer the outflow mask
    outflow_mask_final = np.ones(np.shape(data_cube))
    
##############################################################################
    ### Data input on the console
    
    ## The used sigma-level
    print("Sigma-Level: (default = 3)")
    
    try:
        SIGMA_LEVEL = float(input())
    
    except:
        SIGMA_LEVEL = 3
    
    ## Ask whether the mask can be checked or not
    print("Check the mask? (y/n)")
    
    CHECK_MASK = input()
    
    if CHECK_MASK == "":
        CHECK_MASK == "n"
    
    
##############################################################################
    ### Creating the file and writing its header with pickle to save space
    
    ## Opening/creating the file
    final_outflow = open("%s" %(OUTFLOW_NAME), "wb")
    
    ## Writing SIGMA_LEVEL
    pkl.dump("Sigma-level =\t%.2f\n" %(SIGMA_LEVEL), final_outflow)
    
    ## Writing wheter or not the mask was checked
    pkl.dump("Mask checked = %s (no, if != y or != Y)\n" %(CHECK_MASK), final_outflow)

    
##############################################################################
    ### Iterating over all velocity slices
    
    for i, data_slice in enumerate(data_cube):
    
        ## Progress feedback
        print("Data slice %i of %i" %(i+1, len(data_cube)))
    
        ## Masking all values below the sigma level
        data_slice_asl = np.ma.masked_less(data_slice, SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice))
    
        ## Finding and labeling coherent areas
        labels, nb = ndimage.label(~data_slice_asl.mask)
    
        ## Finding the area label containing the brightest pixel (default of outflow)
        label_max = labels[np.where(data_slice == np.max(data_slice))].max()
    
        ## Masking all values except this area
        data_outflow = np.ma.masked_array(data_slice, mask=(labels != label_max))
    
        ## If turned on, check the mask
        if CHECK_MASK not in ["n", "N"]:
    
            ## Slope describing this region
            sl = ndimage.find_objects(labels==label_max)[0]
            
            edge_color = mcm.get_cmap("summer")(0)
    
            ## Plot the data
            plt.figure(figsize=(12, 12))
    
            # All data above sigma level
            plt.subplot(2, 2, 1)
            plt.imshow(data_slice, cmap="Greys")
            plt.imshow(data_outflow, cmap="summer", zorder=2)
            plt.contourf(data_slice, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()],
                         colors=(edge_color, ), zorder=1)
            plt.title(r'All data above %.1f $\sigma$' %(SIGMA_LEVEL))
            plt.ylabel("Y-axis")
            plt.xlabel("X-axis")
            plt.clim(data_slice_asl.min(), data_slice_asl.max())
    
    
            # Compare the n sigma contours of the neighbouring velocity channels
            plt.subplot(2, 2, 2)
            contour_labels = []     # Empty contour labels
    
            # Contour level of current velocity
            plt.contourf(data_slice, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()], colors="green")
            green_line = mlines.Line2D([], [], color=edge_color,
                          markersize=15, label=r'Above %.1f $\sigma$ level' %(SIGMA_LEVEL))
            contour_labels.append(green_line)
    
            # Try to plot the previous contour levels
            if not i == 0:
                data_slice_previous = data_cube[i-1]
                plt.contour(data_slice_previous, levels=[SIGMA_LEVEL*np.std(data_slice_previous)+np.mean(data_slice_previous)], colors="blue")
                blue_line = mlines.Line2D([], [], color='blue',
                              markersize=15, label=r'Previous %.1f $\sigma$ level' %(SIGMA_LEVEL))
                contour_labels.append(blue_line)
    
            # ... and next slice velocity n sigma levels
            if not i == np.shape(data_cube)[0]-1:
                data_slice_next = data_cube[i+1]
                plt.contour(data_slice_next, levels=[SIGMA_LEVEL*np.std(data_slice_next)+np.mean(data_slice_next)], colors="red")
                red_line = mlines.Line2D([], [], color='red',
                              markersize=15, label=r'Next %.1f $\sigma$ level' %(SIGMA_LEVEL))
                contour_labels.append(red_line)
    
            plt.legend(handles=contour_labels)
            plt.ylabel("Y-axis")
            plt.xlabel("X-axis")
            plt.title(r'Comparison of the %.1f $\sigma$ levels' %(SIGMA_LEVEL))
            plt.xlim(0,np.shape(data_slice)[0])
            plt.ylim(np.shape(data_slice)[1], 0)
    
            # The selected outflow; black frame --> zomed in region
            plt.subplot(2, 2, 3)
            plt.imshow(data_slice, cmap="Greys")
            plt.imshow(data_outflow, cmap="summer", zorder=2)
            plt.contourf(data_outflow, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()],
                         colors=(edge_color, ), zorder=1)
            plt.plot([sl[1].start, sl[1].start, sl[1].stop-1, sl[1].stop-1, sl[1].start], [sl[0].start, sl[0].stop-1, sl[0].stop-1, sl[0].start, sl[0].start], c="black")
            plt.title("Suggested outflow data")
            plt.ylabel("Y-axis")
            plt.xlabel("X-axis")
            plt.clim(data_slice_asl.min(), data_slice_asl.max())
    
            # Zoomed in on the selected outflow
            plt.subplot(2,2,4)
            plt.imshow(data_outflow, cmap="summer")
            plt.title('Cropped outflow data')
            plt.xlim(sl[1].start, sl[1].stop-1)
            plt.ylim(sl[0].stop-1, sl[0].start)
            plt.ylabel("Y-axis")
            plt.xlabel("X-axis")
            plt.clim(data_slice_asl.min(), data_slice_asl.max())
            plt.colorbar()
    
            plt.show()
    
            ## Double checking the outflow region
            print("Is the outflow region valid? (y/n)")
            goon = input()
    
            ## Updating the mask
            outflow_mask = data_outflow.mask
    
            ## If necessary, rework the outflow region
            while goon not in ["y", "Y"]:
    
                ## Resetting the feedback variable (if the selected region shall be handled or not)
                fb = None
    
                ## Ask whether regions shall be added to or removed from the outflow
                print("Add (ad) or remove (rm) regions? Or cancle (ca)?")
    
                ## Input the action
                option = input()
    
                ## Ensure that the entry is correct
                while option not in ["a", "ad", "r", "rm", "ca"]:
    
                    print("Check entry!")
                    print("Add (ad) or remove (rm) regions? Or cancle (ca)?")
    
                    option = input()
    
                ## If one wants to add a region (manuel or with label) to the outflow
                if option in ["a", "ad"]:
    
                    ## Plotting the whole data
                    plt.figure(figsize=(12, 12))
                    plt.contourf(data_slice, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()],
                         colors=(edge_color, ), zorder=1)  
                    plt.imshow(data_slice, cmap="Greys")
                    plt.imshow(data_slice_asl, cmap="summer", zorder=1)
                    plt.title(r"Data above %.1f $\sigma$" %(SIGMA_LEVEL))
                    plt.xlim(0,np.shape(data_slice)[0])
                    plt.ylim(np.shape(data_slice)[1], 0)
                    plt.ylabel("Y-axis")
                    plt.xlabel("X-axis")
                    plt.grid(color="black")
                    plt.colorbar()
    
                    # Plotting previous ...
                    if not i == 0:
                        plt.contour(data_slice_previous, levels=[SIGMA_LEVEL*np.std(data_slice_previous)+np.mean(data_slice_previous)], colors="blue")
    
                    # ... and next slice velocity n sigma levels
                    if not i == np.shape(data_cube)[0]-1:
                        plt.contour(data_slice_next, levels=[SIGMA_LEVEL*np.std(data_slice_next)+np.mean(data_slice_next)], colors="red")
    
    
                    plt.legend(handles=contour_labels[1:])
                    plt.show()
    
                    ## Differ between adding a labeled oder manuel region
                    print("Selcet a label (la) or a region (re)?")
    
                    lor = input()
    
                    ## Ensure that the entry is correct
                    while lor not in ["la", "l", "r", "re"]:
    
                        print("Check entry!")
    
                        print("Selcet a label (la) or a region (re)?")
    
                        lor = input()
    
                    ## If one wants to add a labeled region
                    if lor in ["la", "l"]:
    
                        ## Looping until the area is satisfying (or the progress is cancelled)
                        while fb not in ["y", "ca"]:
    
                            ## Selecting a label or getting help/the n biggest regions
                            print("Select label (n in [%i, %i]) or get the n biggest regions (h):" %(1, labels.max()))
    
                            adn = input()
    
                            ## Selecting labelled area
                            if not adn in ["h", "H"]:
    
                                ## Trying to read out an integer/the area label
                                try:
                                    adn = int(adn)
    
                                except:
                                    adn = -1
    
                                ## Check if the area label is valid
                                if 1 <= adn <= labels.max():
    
                                    ## Masking all data except the selected area
                                    data_outflow_mod = np.ma.masked_array(data_slice, mask=(labels != adn))
    
                                    ## Getting the zoomed in slice
                                    sl_mod = ndimage.find_objects(labels==adn)[0]
    
                                    ## Plot the data
                                    plt.figure(figsize=(12, 6))
                                    
                                    # Plot a large picture
                                    plt.subplot(1, 2, 1)
                                    plt.imshow(data_slice, cmap="Greys")
                                    plt.imshow(data_outflow_mod, cmap="summer")
                                    plt.contour(data_slice, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()],
                                                 colors=(edge_color, ))
                                    plt.contourf(data_outflow_mod, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()],
                                                 colors=(edge_color, ), zorder=1)
                                    plt.plot([sl_mod[1].start, sl_mod[1].start, sl_mod[1].stop-1, sl_mod[1].stop-1, sl_mod[1].start], [sl_mod[0].start, sl_mod[0].stop-1, sl_mod[0].stop-1, sl_mod[0].start, sl_mod[0].start], c="black")
                                    plt.title("Selected data")
                                    plt.xlim(0,np.shape(data_slice)[0])
                                    plt.ylim(np.shape(data_slice)[1], 0)
                                    plt.ylabel("Y-axis")
                                    plt.xlabel("X-axis")
                                    plt.legend(handles=contour_labels[:1])
    
                                    # Plot the zoomed in version
                                    plt.subplot(1, 2, 2)
                                    plt.imshow(data_outflow_mod, cmap="summer")
                                    plt.title('Cropped data of region %i' %(adn))
                                    plt.xlim(sl_mod[1].start, sl_mod[1].stop)
                                    plt.ylim(sl_mod[0].stop, sl_mod[0].start)
                                    plt.ylabel("Y-axis")
                                    plt.xlabel("X-axis")
                                    plt.colorbar()
    
                                    plt.show()
    
                                    ## Getting feedback on the outflow data
                                    print("Add this one? (y/n) Or cancle (ca)?")
                                    fb = input()
    
                                else:
    
                                    print("Invalid entry. Try again.")
    
                            ## Else get the labels of the n largest regions
                            else:
                                print("Number of largest ensemble (n in [%i; %i]):" %(1, labels.max()))
    
                                ## Read and check the entry
                                try:
                                    nle = int(input())
    
                                except:
                                    nle = -1
    
                                ## Ensure that the entry is correct
                                while not 1 <= nle <= labels.max():
    
                                    print("Invalid entry. Try again.")
    
                                    print("Number of largest ensemble (n in [%i; %i]):" %(1, labels.max()))
    
                                    ## Read and check the entry
                                    try:
                                        nle = int(input())
    
                                    except:
                                        nle = -1
    
    
                                ## Get the size of each labeled region
                                labels_unique, labels_counts = np.unique(labels, return_counts=True)
    
                                ## Sort the regions by size
                                labels_unique_sorted = labels_unique[1:][(-labels_counts[1:]).argsort()]
    
                                ## Print the result
                                print("The %i biggest regions are:\t%a\n" %(nle, labels_unique_sorted[:nle]))
    
                        ## If the action is not cancelled ...
                        if not fb == "ca":
    
                            ## ... update the mask
                            outflow_mask = np.logical_and(outflow_mask, data_outflow_mod.mask)
    
                    ## If one wants to add a manual region/shape
                    elif lor in ["re", "r"]:
    
                        ## Adjust the area until it is satisfying or the action is cancelled
                        while fb not in ["y", "ca"]:
    
                            ## Ask for an implemented shape
                            print("Add a rectangle (r) or polygon (p)? Or cancle (ca)?")
    
                            form = input()
    
                            ## Ensure that the entry is correct
                            while form not in ["r", "p", "ca"]:
    
                                print("Check entry!")
    
                                print("Add a rectangle (r) or polygon (p)? Or cancle (ca)?")
    
                                form = input()
    
                            ## For a rectangle
                            if form == "r":
    
                                ## Reset the y components as they are needed for the validation check
                                y1 = None
                                y2 = None
    
                                ## Getting upper left point (and ensuring that it is correct)
                                while y1 is None:
    
                                    print("Its upper left corner (x, y):")
    
                                    uple = input()
    
                                    ## Try to read out the point
                                    try:
                                        x1 = int(uple.split(",")[0])
                                        y1 = int(uple.split(",")[1])
    
                                    except:
                                        print("Check entry!")
    
                                ## Getting lower right point (and ensuring that it is correct)
                                while y2 is None:
    
                                    print("Its lower right corner (x, y):")
    
                                    lori = input()
    
                                    ## Try to read out the point
                                    try:
                                        x2 = int(lori.split(",")[0])
                                        y2 = int(lori.split(",")[1])
    
                                    except:
                                        print("Check entry!")
    
    
                                ## Creating a polygram out of these edge points
                                poly_verts = [(x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1)]
    
    
                            ## For a polygram
                            elif form == "p":
    
                                ## Empty list for polygram edges
                                poly_verts = []
    
                                ## Go back to start position?
                                stpo = "n"
    
                                ## Adding edge points as long as one does not want to return to the staring point
                                while stpo != "y":
    
                                    ## Adding corner points
                                    print("Add corner point (x, y) or go back to starting point (s):")
    
                                    corn = input()
    
                                    ## Ensure that the entry is correct
                                    try:
    
                                        x1 = int(corn.split(",")[0])
                                        y1 = int(corn.split(",")[1])
    
                                        ## Adding the point to the edge list
                                        poly_verts.append((x1, y1))
    
                                    except:
    
                                        ## Check if one wants to go back to the starting point ...
                                        if corn in ["s", "S"]:
    
                                            ## ... and add it to the polygram list
                                            poly_verts.append(poly_verts[0])
    
                                            stpo = "y" # Exit the while loop
    
                                        else:
                                            print("Check entry!")
                                            
                            # if (np.min(poly_verts, axis=0) == np.max(poly_verts, axis=0)).any():
                            #     print("The shape is one dimensional. Cancel action.")
                            #     fb = "ca"
    
                            ## Cancle adding data
                            elif form == "ca":
                                fb = "ca"
    
                            ## Handling the polynom if action is not cancelled
                            if fb != "ca":
    
                                ## Create a mask matching the figure
                                img = Image.new('L', np.shape(data_slice), 0)
                                ImageDraw.Draw(img).polygon(poly_verts, outline=1, fill=1)
                                form_mask = np.array(img)
                                
                                ## Combining the form mask and the data-slice-above-sigma-level mask
                                form_mask = ~np.logical_and(form_mask, ~data_slice_asl.mask)
    
                                ## Applying the mask
                                data_outflow_mod = np.ma.masked_array(data_slice, mask=form_mask)
    
                                ## Try to get the zoomed in slice and data and expanding it if its width is one
                                try:
                                    sl_mod = ndimage.find_objects(~data_outflow_mod.mask)[0]
                                    
                                    if sl_mod[1].start == sl_mod[1].stop-1:
                                        sl_mod = (sl_mod[0], slice(sl_mod[1].start-5, sl_mod[1].stop+5, None))
                                        
                                    if sl_mod[0].start == sl_mod[0].stop-1:
                                        sl_mod = (slice(sl_mod[0].start-5, sl_mod[0].stop+5, None), sl_mod[1])
                                    
                                except:
                                    sl_mod = None
                                    
    
                                ## Plotting the suggested outflow
                                plt.figure(figsize=(12, 6))
    
                                # Plot the whole data
                                plt.subplot(1, 2, 1)
                                plt.imshow(data_slice, cmap="Greys")
                                plt.contour(data_slice, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice)], colors=(edge_color, ))
                                
                                if sl_mod is not None:
                                    plt.imshow(data_outflow_mod, cmap="summer")
                                    plt.contourf(data_outflow_mod, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()],
                                                 colors=(edge_color, ), zorder=1)
                                    
                                plt.plot(np.array(poly_verts).T[0],np.array(poly_verts).T[1], c="black")
    
                                plt.title("Selected data")
                                plt.ylabel("Y-axis")
                                plt.xlabel("X-axis")
                                plt.xlim(0,np.shape(data_slice)[0])
                                plt.ylim(np.shape(data_slice)[1], 0)
                                plt.grid(color="black")
                                plt.legend(handles=contour_labels[:1])
    
                                # Plot the zoomed in version
                                plt.subplot(1, 2, 2)
                                plt.title('Cropped data')
                                
                                if sl_mod is not None:
                                    plt.imshow(data_outflow_mod, cmap="summer", zorder=1)
                                    plt.xlim(sl_mod[1].start, sl_mod[1].stop-1)
                                    plt.ylim(sl_mod[0].stop-1, sl_mod[0].start)
                                    plt.plot(np.array(poly_verts).T[0],np.array(poly_verts).T[1], c="black")
                                    plt.colorbar()
    
                                else:
                                    plt.plot()
                                    plt.xlim(-.05, .05)
                                    plt.ylim(.05, -.05)
                                    print("Warning: There is no outflow data inside this region!")
    
    
                                plt.ylabel("Y-axis")
                                plt.xlabel("X-axis")
                                plt.grid(color="black")
    
                                plt.show()
    
                                ## Double check the selected region
                                print("Add this region? (y/n) Or cancle (ca)?")
                                fb = input()
    
                                if fb in ["y", "Y"]:
                                    outflow_mask = np.logical_and(outflow_mask, data_outflow_mod.mask)
    
                ## If one wants to remove data
                elif option in ["r", "rm"]:
    
                    if False in data_outflow.mask:
    
                        ## Plotting the whole data
                        plt.figure(figsize=(12, 12))
                        plt.imshow(data_slice, cmap="Greys")
                        plt.contourf(data_slice, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()],
                                     colors=(edge_color, ), zorder=1)
                        plt.imshow(data_outflow, cmap="summer", zorder=1)
                        plt.title(r"Data above %.1f $\sigma$" %(SIGMA_LEVEL))
                        plt.xlim(0,np.shape(data_slice)[0])
                        plt.ylim(np.shape(data_slice)[1], 0)
                        plt.ylabel("Y-axis")
                        plt.xlabel("X-axis")
                        plt.grid(color="black")
                        plt.colorbar()
    
                        # Plotting previous ...
                        if not i == 0:
                            plt.contour(data_slice_previous, levels=[SIGMA_LEVEL*np.std(data_slice_previous)+np.mean(data_slice_previous)], colors="blue")
    
                        # ... and next slice velocity n sigma levels
                        if not i == np.shape(data_cube)[0]-1:
                            plt.contour(data_slice_next, levels=[SIGMA_LEVEL*np.std(data_slice_next)+np.mean(data_slice_next)], colors="red")
    
                        plt.legend(handles=contour_labels[1:])
                        plt.show()
    
    
                        ## Differ between adding a labeled oder manuel region
                        print("Selcet a label (la), a region (re) or remove all (al)? Or chancle (ca)?")
    
                        lor = input()
    
                        ## Ensure that the entry is correct
                        while lor not in ["la", "l", "r", "re", "a", "al", "all", "c", "ca"]:
    
                            print("Check entry!")
    
                            print("Selcet a label (la), a region (re) or remove all (al)? Or chancle (ca)?")
    
                            lor = input()
    
                        ## Cancel action if one wants to
                        if lor in ["c", "ca"]:
                            fb = "ca"
                            
                        while fb not in ["y", "Y", "ca"]:
    
                            ## Listing all used labels
                            label_list = np.unique(np.ma.masked_array(labels, mask=data_outflow.mask))[:-1].data
    
                            ## Cancel action if there is no data left
                            if len(label_list) == 0:
                                print("Can not remove a label from an empty label list. Action cancelled.")
                                fb = "ca"
                            
                            ## If one wants to remove a label
                            elif lor in ["la", "l"] and len(label_list) != 0:
    
                                print("Select label (n in %a)" %(label_list))
    
                                ## Try to read out wich label one wants to remove
                                try:
                                    rmn = int(input())
    
                                except:
                                    rmn = -1
    
                                ## If one can remove this label
                                if rmn in label_list:
    
                                    ## Getting label data
                                    data_outflow_mod = np.ma.masked_array(data_slice, mask=(labels != rmn))
    
                                    ## The zoomed in slice
                                    sl_mod = ndimage.find_objects(labels==rmn)[0]
    
                                    ## Plotting this region
                                    plt.figure(figsize=(12, 6))
    
                                    # Plot the selected data and the contor levels
                                    plt.subplot(1, 2, 1)
                                    plt.imshow(data_slice, cmap="Greys")
                                    plt.imshow(data_outflow_mod, cmap="summer")
                                    plt.contourf(data_outflow_mod, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()],
                                                 colors=(edge_color, ), zorder=1)
                                    plt.plot([sl_mod[1].start, sl_mod[1].start, sl_mod[1].stop-1, sl_mod[1].stop-1, sl_mod[1].start], [sl_mod[0].start, sl_mod[0].stop-1, sl_mod[0].stop-1, sl_mod[0].start, sl_mod[0].start], c="black")
                                    plt.contour(data_slice, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice)], colors=(edge_color, ))
                                    plt.title("Selected data")
                                    plt.xlim(0,np.shape(data_slice)[0])
                                    plt.ylim(np.shape(data_slice)[1], 0)
                                    plt.legend(handles=contour_labels[:1])
    
                                    # Plot the zoomed in region
                                    plt.subplot(1, 2, 2)
                                    plt.imshow(data_outflow_mod, cmap="summer")
                                    plt.title('Cropped data')
                                    plt.xlim(sl_mod[1].start, sl_mod[1].stop-1)
                                    plt.ylim(sl_mod[0].stop-1, sl_mod[0].start)
                                    plt.colorbar()
    
                                    plt.show()
                                    
                                    ## Double check if one wants to remove this region
                                    print("Remove this one? (y/n) Or cancle (ca)?")
                                    fb = input()
    
                                else:
    
                                    print("Entry is not in labels.")
    
                            ## If one wants to remove a manual region/shape
                            elif lor in ["re", "r"]:
    
                                ## Adjust the area until it is satisfying or the action is cancelled
                                while fb not in ["y", "ca"]:
    
                                    ## Ask for an implemented shape
                                    print("Remove a rectangle (r) or polygon (p)? Or cancle (ca)?")
    
                                    form = input()
    
                                    ## Ensure that the entry is correct
                                    while form not in ["r", "p", "ca"]:
    
                                        print("Check entry!")
    
                                        print("Remove a rectangle (r) or polygon (p)? Or cancle (ca)?")
    
                                        form = input()
    
                                    ## For a rectangle
                                    if form == "r":
    
                                        ## Reset the y components as they are needed for the validation check
                                        y1 = None
                                        y2 = None
    
                                        ## Getting upper left point (and ensuring that it is correct)
                                        while y1 is None:
    
                                            print("Its upper left corner (x, y):")
    
                                            uple = input()
    
                                            ## Try to read out the point
                                            try:
                                                x1 = int(uple.split(",")[0])
                                                y1 = int(uple.split(",")[1])
    
                                            except:
                                                print("Check entry!")
    
                                        ## Getting lower right point (and ensuring that it is correct)
                                        while y2 is None:
    
                                            print("Its lower right corner (x, y):")
    
                                            lori = input()
    
                                            ## Try to read out the point
                                            try:
                                                x2 = int(lori.split(",")[0])
                                                y2 = int(lori.split(",")[1])
    
                                            except:
                                                print("Check entry!")
    
    
                                        ## Creating a polygram out of these edge points
                                        poly_verts = [(x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1)]
    
    
                                    ## For a polygram
                                    elif form == "p":
    
                                        ## Empty list for polygram edges
                                        poly_verts = []
    
                                        ## Go back to start position?
                                        stpo = "n"
    
                                        ## Adding edge points as long as one does not want to return to the staring point
                                        while stpo != "y":
    
                                            ## Adding corner points
                                            print("Add corner point (x, y) or go back to starting point (s):")
    
                                            corn = input()
    
                                            ## Ensure that the entry is correct
                                            try:
    
                                                x1 = int(corn.split(",")[0])
                                                y1 = int(corn.split(",")[1])
    
                                                ## Adding the point to the edge list
                                                poly_verts.append((x1, y1))
    
                                            except:
    
                                                ## Check if one wants to go back to the starting point ...
                                                if corn in ["s", "S"]:
    
                                                    ## ... and add it to the polygram list
                                                    poly_verts.append(poly_verts[0])
    
                                                    stpo = "y" # Exit the while loop
    
                                                else:
                                                    print("Check entry!")
                                                    
                                    ## Cancle removing data
                                    elif form == "ca":
                                        fb = "ca"
    
                                    ## Handling the polynom if action is not cancelled
                                    if fb != "ca":
    
                                        ## Create a mask matching the figure
                                        img = Image.new('L', np.shape(data_slice), 0)
                                        ImageDraw.Draw(img).polygon(poly_verts, outline=1, fill=1)
                                        form_mask = np.array(img)
    
                                        ## Combining the form mask and the data-slice-above-sigma-level mask
                                        form_mask = ~np.logical_and(form_mask, ~data_outflow.mask)
    
                                        ## Applying the mask
                                        data_outflow_mod = np.ma.masked_array(data_slice, mask=form_mask)
                                        
                                        ## Try to get the zoomed in slice and data and expanding it if its width or length is one
                                        try:
                                            sl_mod = ndimage.find_objects(~data_outflow_mod.mask)[0]
                                            
                                            if sl_mod[1].start == sl_mod[1].stop-1:
                                                sl_mod = (sl_mod[0], slice(sl_mod[1].start-5, sl_mod[1].stop+5, None))
                                                
                                            if sl_mod[0].start == sl_mod[0].stop-1:
                                                sl_mod = (slice(sl_mod[0].start-5, sl_mod[0].stop+5, None), sl_mod[1])
                                                
                                        except:
                                            sl_mod = None
    
    
                                        ## Plotting the suggested outflow
                                        plt.figure(figsize=(12, 6))
    
                                        # Plot the whole data
                                        plt.subplot(1, 2, 1)
                                        plt.imshow(data_slice, cmap="Greys")
                                        plt.contour(data_slice, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice)], colors=(edge_color, ))
          
                                        if sl_mod is not None:
                                            plt.imshow(data_outflow_mod, cmap="summer")
                                            plt.contourf(data_outflow_mod, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()],
                                                         colors=(edge_color, ), zorder=1)
                                            
                                        plt.plot(np.array(poly_verts).T[0],np.array(poly_verts).T[1], c="black")
                                        plt.title("Selected data")
                                        plt.xlim(0,np.shape(data_slice)[0])
                                        plt.ylim(np.shape(data_slice)[1], 0)
                                        plt.ylabel("Y-axis")
                                        plt.xlabel("X-axis")
                                        plt.grid(color="black")
                                        plt.legend(handles=contour_labels[:1])
    
                                        # Plot the zoomed in version
                                        plt.subplot(1, 2, 2)
                                        plt.title('Cropped data')
    
                                        if sl_mod is not None:
                                            plt.imshow(data_outflow_mod, cmap="summer", zorder=1)
                                            plt.xlim(sl_mod[1].start, sl_mod[1].stop-1)
                                            plt.ylim(sl_mod[0].stop-1, sl_mod[0].start)
                                            plt.plot(np.array(poly_verts).T[0],np.array(poly_verts).T[1], c="black")
                                            plt.colorbar()
    
                                        else:
                                            plt.plot()
                                            plt.xlim(-.05, .05)
                                            plt.ylim(.05, -.05)
                                            
                                            print("Warning: There is no outflow data inside this region!")
    
                                        plt.ylabel("Y-axis")
                                        plt.xlabel("X-axis")
                                        plt.grid(color="black")
    
                                        plt.show()
    
                                        ## Double check the selected region
                                        print("Remove this region? (y/n) Or cancle (ca)?")
                                        fb = input()
    
    
                            ## If one wants to remove all values
                            if lor in ["a", "al", "all"]:
    
                                ## Double check the entry
                                print("Do you really want to remove the whole data? (y/n) Or cancle (ca)?")
    
                                fb = input()
    
                                if fb in ["y", "Y"]:
    
                                    ## Filling the mask with 1 --> True
                                    outflow_mask.fill(1)
                                    data_outflow_mod = np.ma.masked_array(data_outflow, mask=outflow_mask)
    
                    else:
                        print("Can not remove any data as the array is completly masked.")
                        fb = "ca"
    
                    if fb != "ca":
                        ## Updating the outflow mask
                        outflow_mask = ~np.logical_and(~outflow_mask, data_outflow_mod.mask)
    
                ## Applying the new mask
                data_outflow = np.ma.masked_array(data_slice, mask=outflow_mask)
    
                ## Finging all unmasked values inside the data to croop it
                outflow_mask_false = np.where(outflow_mask == False)
                         
                ## Try to get the zoomed in slice and data and expanding it if its width or length is one
                try:
                    sl = (slice(outflow_mask_false[0].min(),outflow_mask_false[0].max()), slice(outflow_mask_false[1].min(), outflow_mask_false[1].max()))
                    
                    if sl[1].start == sl[1].stop-1:
                        sl = (sl[0], slice(sl[1].start-5, sl[1].stop+5, None))
                        
                    if sl[0].start == sl[0].stop-1:
                        sl = (slice(sl[0].start-5, sl[0].stop+5, None), sl[1])
                  
                except:
                    sl = None
    
                ## Plot updated outflow region
                plt.figure(figsize=(12, 12))
    
                # Plot all data
                plt.subplot(2, 2, 1)
                plt.imshow(data_slice, cmap="Greys")
                plt.imshow(data_slice_asl, cmap="summer", zorder=2)
                plt.contourf(data_slice, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()],
                             colors=(edge_color, ), zorder=1)
                plt.title(r'All data above %.1f $\sigma$' %(SIGMA_LEVEL))
                plt.ylabel("Y-axis")
                plt.xlabel("X-axis")
                plt.clim(data_slice_asl.min(), data_slice_asl.max())
    
                # Compare the n sigma contours of the neighbouring velocity channels
                plt.subplot(2, 2, 2)
    
                # Contour level of current velocity
                plt.contourf(data_slice, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()], colors=(edge_color, ))
    
                # Try to plot the previous contour levels
                if not i == 0:
                    plt.contour(data_slice_previous, levels=[SIGMA_LEVEL*np.std(data_slice_previous)+np.mean(data_slice_previous)], colors="blue")
    
                # ... and next slice velocity n sigma levels
                if not i == np.shape(data_cube)[0]-1:
                    plt.contour(data_slice_next, levels=[SIGMA_LEVEL*np.std(data_slice_next)+np.mean(data_slice_next)], colors="red")
    
                plt.legend(handles=contour_labels)
                plt.ylabel("Y-axis")
                plt.xlabel("X-axis")
                plt.title(r'Comparison of the %.1f $\sigma$ levels' %(SIGMA_LEVEL))
                plt.xlim(0,np.shape(data_slice)[0])
                plt.ylim(np.shape(data_slice)[1], 0)
    
                # Selected outflow pixles; black frame --> zomed in region
                plt.subplot(2, 2, 3)
                plt.imshow(data_slice, cmap="Greys")
                plt.ylabel("Y-axis")
                plt.xlabel("X-axis")
    
                if sl is not None:
                    plt.plot([sl[1].start, sl[1].start, sl[1].stop, sl[1].stop, sl[1].start], [sl[0].start, sl[0].stop, sl[0].stop, sl[0].start, sl[0].start], c="black")
                    plt.imshow(data_outflow, cmap="summer", zorder=2)
                    plt.contourf(data_outflow, levels=[SIGMA_LEVEL*np.std(data_slice)+np.mean(data_slice), data_slice.max()],
                                 colors=(edge_color, ), zorder=1)
                    
                plt.clim(data_slice_asl.min(), data_slice_asl.max())
                plt.title("Suggested outflow data")
    
                # Zoomed in on the selected outflow
                plt.subplot(2,2,4)
    
                if sl is not None:
                    plt.imshow(data_outflow, cmap="summer", zorder=2)
                    plt.xlim(sl[1].start, sl[1].stop)
                    plt.ylim(sl[0].stop, sl[0].start)
    
                else:
                    plt.plot()
                    plt.xlim(-.05, .05)
                    plt.ylim(.05, -.05)
                    print("Warning: There is no outflow data inside this region!")
    
    
                plt.clim(data_slice_asl.min(), data_slice_asl.max())
                plt.title('Cropped outflow data')
                plt.ylabel("Y-axis")
                plt.xlabel("X-axis")
                plt.colorbar()
    
                plt.show()
    
    
                print("Is the outflow region valid? (y/n)")
                goon = input()
    
        ## Buffering the mask slice
        outflow_mask_final[i] = data_outflow.mask
    
##############################################################################
    ### Writing data to file
    pkl.dump(outflow_mask_final, final_outflow)

    ### Closing file
    final_outflow.close()


"""
The following code can be used to read the mask:

import numpy as np
import pickle as pkl


## Open the file containing the mask (reading is linewise)
outflow_file = open(__FILE_NAME__, "rb")

## Reading out the sigma level
SIGMA_LEVEL = float(pkl.load(outflow_file).split()[2])

## Jumping unimportand line
pkl.load(outflow_file)

## Reading the array
outflow_mask = pkl.load(outflow_file).reshape(X1, X2, X3)
"""
