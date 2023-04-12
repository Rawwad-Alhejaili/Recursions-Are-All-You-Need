#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I have written this function to make it much quicker to show high quality
figures. I find that showing images using plt.imshow to be fairly frustrating
because the defaults do not suit my needs at all, and adjusting them every
single time does get annoying quickly. As such, I decided to write
this function to cut down on the tedium. Surprisingly, this function makes
showing images much more convenient (to me). Below I will write a summary that 
includes most of the capabilities provided here.

- Automatically applies cmap='seismic_r' to one channel images (configurable)
- Removes the axes from the image
- Works with tensors (by moving them to the cpu and detaching the gradients)
- Works with 4D arrays (it uses the first sample for convenience)
- Automatically moves the channel axis position to the last axis
- Sets the DPI to 300 by default (configurable)
- Adds a colorbar to the right (can be toggled on or off from the arguments)
- Accepts multiple images and shows them on a specified grid (e.g. 2x2 vs 1x4)
- Titles can be assigned to multiple images (optional)
- All images are shown in a unified range. Range of the last image is used 
  by default (configurable via the rang argument)
- Can Automatically clips the minimum and maximum 1 percentile of the image 
  (very useful for seismic images)
- Padding between the images is configurable (default = 0.7)
- The figure can be set to be transparent (hope that I spelled that right)
- Other plt.imshow arguments can be passed as named arguments **kwargs.
  For example, assume we want to change the aspect ratio of the shown
  image. This can be modified by using the "aspect" argument in plt.imshow.
  We could pass this to our function by kwargs as:
      kwargs = {"aspect": 1/3}
  Where the key is the name of the argument and the value is... its value (surprise!).
  Or, you could simply add aspect=1/3 in the arguments of this function 
  without the use of a dictionary. I guess the explanation here is pretty
  convoluted ):
  
What to work on next:
    - Add support for xlabels (my attempts are buggy)
    - Add suptitile (global title) for the figure (currently, the padding is too high)
    - Write the function as a class instead, so you can define your defaults 
      only once in the code. For example, if you work with cmap='gray', then
      you will initialize the class with cmap='gray', so next time you call
      the imshow method, it will use cmap='gray' by default (simple to do)
    - Add the option to normalize each image individually (simple to do)
    - Add more informative messages (with the ability to suppress them)
    - Add the offset and time axes.
    
Finally, a weird tip. If you create a subplot with a grid = (3,3) for example,
but you don't need the image in the middle of the figure for some reason, 
then it can be deleted by feeding the location of the middle image with a 
constant number (or an image with zero standard deviation). The function will 
recognize the uselessness of the image and will discard it from the image grid 
IF THE ARGUMENT `ignoreZeroStd=True`.
Again, this is too convoluted, so I will simply stop trying to explain :)

@author: Rawwad Alhejaili
"""

import numpy as np
import torch
import matplotlib as mpl
# mpl.interactive(False)  # MIGHT prevent memory leak in jupyter and Spyder IDE (IT DID NOT!)
import matplotlib.pyplot as plt
# from matplotlib import ticker
from mpl_toolkits.axes_grid1 import ImageGrid
import warnings
import torch.fft as fft
from scipy.fft import fft2, fftshift
# fft2tor = lambda x, k: 20*torch.log10(torch.finfo(x.dtype).eps+torch.abs(fft.fftshift(fft.fft2(x, (x.shape[-2]*k, x.shape[-1]*k)), (-2,-1)))).cpu().detach()
fft2tor = lambda x, k: torch.finfo(x.dtype).eps \
                     + torch.abs(fft.fftshift(fft.fft2(x, (x.shape[-2]*k, x.shape[-1]*k)), (-2,-1))).cpu().detach()
fft2np  = lambda x, k: np.finfo(x.dtype).eps \
                     + np.abs(fftshift(fft2(x, (x.shape[-2]*k, x.shape[-1]*k)), (-2,-1)))
decibal = lambda x: 20*np.log10(x)

def imshow2(I, 
            title          = None, 
            grid           = (1,1), 
            colorbar       = True, 
            cbar_ticks     = 11, 
            cbar_label     = None,
            aspect         = 1, 
            cmap           = None, 
            pad            = 0.7, 
            plotsize       = (5,5), 
            rang           = 'global', 
            rangZeroCenter = False, 
            dpi            = 300, 
            figTransparent = False, 
            fontsize       = 14, 
            alphabet       = False, 
            ignoreZeroStd  = False, 
            clip           = 0, 
            t              = None,
            offset         = None,
            fft            = False,
            fft_factor     = 3,
            dB             = None,
            transform      = None,
            **kwargs):
    '''
    Parameters
    ----------
    I : TYPE = List
        A list that includes all images to be shown.
    title : TYPE = List of strings, optional
        Includes the titles of the images in I. The default is none.
    grid : TYPE = tuple, list, or array with shape=(2,)
        Assigns the grid layout of subplots. The default is (1,1) in case a single image is used.
    colorbar : TYPE = bool, optional
        Specifies whether to add a colorbar or not. The default is True.
    cbar_ticks : TYPE = int, optional
        Controls the number of ticks in the color bar (this is a bit buggy)
    cbar_label : TYPE = str, optional
        Adds a label to the colorbar
    aspect : TYPE = float, optional
        Sets the aspect ratio of the plots. The default is 1.
    cmap : TYPE: str, optional
        Specifies the colormap to be used for one channel images. The default is 'seismic'.
    pad : TYPE = float, optional
        Adjusts the padding between subplots. The default is 0.7.
    plotsize : TYPE = tuple, list, or array with shape=(2,)
        Adjusts the SUBPLOT size (seems to be buggy). The default is (5,5).
    rang : TYPE = tuple=(vmin,vmax), or int (index of argument I), optional
        When the type is a tuple, list, or an array with shape (2,), then the
        range of the color map will be rang[0] to rang[1].
        When the type is an integer, then the normalization will be performed 
        based on the range of I[rang]. 
        Note: the range will still be clipped (removing the lowest and highest 1%)
        By default, the range=-1 (last image in argument I) is used for the
        color map. Quite useful for comparing multiple image.
    rangZeroCenter : TYPE bool, optional
        If true, the center of the range will be zero. Useful when finding
        the difference between two images. 
        Note that if rang has an explicit vmin and vmax, then it will override 
        this argument (setting it to False).
        The default is False.
    dpi : TYPE = int, optional
        Sets the dpi of the figure. The default is 300.
    figTransparent : TYPE = bool, optional
        Sets the transparency of the figure. The default is False.
    fontsize : TYPE = int, optional
        Sets the font size of the title. The default is 14.
    alphabet : TYPE = bool, optional (BUGGY for multiple rows)
        Adds letter labels to the subplots. The default is False.
    ignoreZeroStd : TYPE = bool, optional
        If true, images with zero standard deviation will not be drawn.
        The default is True.
    clip : TYPE = bool, optional
        Clips the minimum and maximum x% of the data (based on the frequency 
        histogram). The default is 0 (no clip:).
    **kwargs : TYPE = who knows :)?
        Named arguments to be passed to plt.imshow. It can be passed either as
        individual named arguments or as a dictionary containing multiple
        named arguments.
        

    Returns
    -------
    Displays the image (or images)

    '''
    
# =============================================================================
# Fixing issues that could arise from unexpected input types
# =============================================================================
    # Check if the input is a list, and fix it if it wasn't
    if type(I) != type([]):
        I = [I]  #Fixes the below for loops
    
    if fft:
        try:
            I = [fft2tor(Im, fft_factor) for Im in I]
        except:
            I = [fft2np(Im, fft_factor) for Im in I]
        if dB is None:
            I = [decibal(Im) for Im in I]
    
    if dB:
        I = [decibal(Im) for Im in I]
        
    if transform is None:
        transform = [lambda x: x for i in range(len(I))]
        
    elif type(transform) != type([]):
        transform = [transform for i in range(len(I))]
    
    I = [transform[i](Im) for i, Im in enumerate(I)]
    
    if title is None:  #To avoid errors (didn't troubleshoot it yet)
        title = ['' for i in range(len(I))]
    elif type(title) != type([]):  #if title was not a list
        title = [title]  #Fixes the below for loops
    
    if len(title) < len(I):
        for i in range(len(I) - len(title)):
            title.append('')
        error = \
'''The number of titles is less than the number of images!
To avoid errors, empty titles were added to the other images'''
        warnings.warn(error, UserWarning)
    
    # Check if the grid is too small to fit the input images
    if len(I) > grid[0]*grid[1]:
        error = \
'''The chosen grid cannot fit the provided images!
To mitigate this, all images will be displayed in a single row.

Note: If you wish to add an empty image at a certain location in the grid, then 
add a constant number there (or an image with zero standard deviation)\n'''
        grid = (1, len(I))
        # raise ValueError(error)
        # I may raise a warning instead and fix this myself (I did)
        warnings.warn(error, UserWarning)
    elif len(I) == 1:
        grid = (1,1)  #if the user entered a single image, then use this grid
    
    if t is None or offset is None:
        # print(0)
        disable_Axes = True
        offset = [np.linspace(0,1,I[idx].shape[-1]) for idx in range(len(I))]
        t      = [np.linspace(0,I[idx].shape[-2]/I[idx].shape[-1],I[idx].shape[-2]) for idx in range(len(I))]
    elif type(t) != type([]): 
        # print(1)
        disable_Axes = False
        t      = [t.reshape(-1).astype(float)      for i in range(len(I))]
        offset = [offset.reshape(-1).astype(float) for i in range(len(I))]
    else:
        # print(2)
        disable_Axes = True
        
    if fft:
        t = [np.linspace(-0.5, 0.5, Im.shape[-2])/(tim[1]-tim[0]) for Im, tim in zip(I,t)]
        offset = [np.linspace(-0.5, 0.5, Im.shape[-1])/(off[1]-off[0]) for Im, off in zip(I,offset)]
        # print(t[0].shape)
        # print(offset[0].shape)
    # print(t)
    # print('\n\n\n')
    # offset = [offset for i in range(len(I))]
    # if (t is None and not any(t)) and (offset is None and not any(offset)):
    #     disable_Axes = True
    #     t = [np.linspace(0,1,I[idx].shape[-2]) for idx in range(len(I))]
    #     offset = [np.linspace(0,1,I[idx].shape[-1]) for idx in range(len(I))]
    # else:
    #     disable_Axes = False
    #     # print('False')
    # # Check if `t` is a list
    # if type(t) != type([]):
    #     t = [t]  #Fixes the below for loops
    
    # # Check if `offset` is a list
    # if type(offset) != type([]):
    #     offset = [offset]  #Fixes the below for loops
        
    # ---------------------------------------------------------------------
    # Fix both the `t` and `offset` axes
    # ---------------------------------------------------------------------
    for idx in range(len(I)):
        # # Make sure both arguments are one-dimensional
        # try:
        #     t[idx] = t[idx].astype('float').reshape(-1)
        # except:
        #     t[idx] = np.arange(I[idx].shape[-2])
        # try:
        #     offset[idx] = offset[idx].astype('float').reshape(-1)
        # except:
        #     offset[idx] = np.arange(I[idx].shape[-1])
        
        # If the size of the arguments don't align with the image, 
        # then fix them
        if t[idx].shape[0] != I[idx].shape[-2]:
            error = \
f'''The time axis at index={idx} is not of the exact same size as the 
corresponding image. Therefore, np.arange will be used to avoid errors.'''
            warnings.warn(error, UserWarning)
            t[idx] = np.arange(I[idx].shape[-2])
        if offset[idx].shape[0] != I[idx].shape[-1]:
            error = \
f'''The offset axis at index={idx} is not of the exact same size as the 
corresponding image. Therefore, np.arange will be used to avoid errors.'''
            warnings.warn(error, UserWarning)
            offset[idx] = np.arange(I[idx].shape[-1])
# =============================================================================
# Create the image grid
# =============================================================================
    # Create a figure which scales in size with respect to the grid size.
    # This makes the text size to the "plot" size stay consistent.
    fig = plt.figure(dpi=dpi, 
                     figsize=(plotsize[0]*grid[1],plotsize[1]*grid[0]))
                     # constrained_layout=True)
    
    # Below is a previous attempt at a custom image grid (consider it as a scrapyard)
    # fig, ax = plt.subplots(grid[0], 
    #                        grid[1], 
    #                        plotsize=(plotsize[0]*grid[1],plotsize[1]*grid[0]), 
    #                        dpi=200)
                            # constrained_layout=True)
    # The figure will have fixed width and its height will be adjusted
    # according to the aspect ratio. Note that it is assumed that all images
    # have the same aspect ratio
    # plt.figure(dpi=300, plotsize=(5.5*grid[1],5.5*(H/W)*grid[0]))
    # fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0, hspace=0)
    
    # Create the image grid and include the colorbar if the user so desires
    if colorbar:
        grid = ImageGrid(fig, 111,  # as in plt.subplot(111) (I don't understand this)
                 nrows_ncols=(grid[0],grid[1]),
                 axes_pad=pad,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=pad,
                 )
    else:
        grid = ImageGrid(fig, 111,
                     nrows_ncols=(grid[0],grid[1]),
                     share_all=True,
                     axes_pad=pad)
    
    if figTransparent:
        fig.patch.set_visible(False)
        
# =============================================================================
# Set the range
# =============================================================================
    
    # Preallocation
    n = len(I)
    minv = np.empty(n)
    maxv = np.empty(n)
    
    # I "guess" that the below for loop is a speed bottle neck, because when a
    # 4D tensor is used that has multiple samples, the function becomes very
    # slow. It is likely that converting all elements to a numpy array can
    # take a lot of time to be completed. Another problem is that this specific
    # for loop is found again in the next for loop, so this is not a bad
    # oppotunity for optimization, even if its impact will rarely be seen
    for i, Im in enumerate(I):
        # Convert the image to a numpy array (in case it was a tensor)
        try:
            if torch.is_tensor(Im):
                Im = Im.detach().float()
                if Im.is_cuda:
                    Im = Im.cpu()
                Im = Im.numpy()
            else:
                Im = Im.astype('float')
        except:
            error = \
f'''The image at index={i} CANNOT be converted to a numpy array.
As such, it will be skipped'''
            warnings.warn(error, UserWarning)
            continue  #Skip to the next image
        
        if clip != 0:
            # The below clips the range (VERY useful for seismic images)
            minv[i] = np.percentile(Im,     clip)  #To set the minimum value of the image
            maxv[i] = np.percentile(Im, 100-clip)  #To set the maximum value of the image
        else:
            # Find the minimum and maximum values of ALL images
            minv[i] = Im.min()  #To set the minimum value of the image
            maxv[i] = Im.max()  #To set the maximum value of the image
            
        if rangZeroCenter:
            # Ensuring the zero value is exactly at the center of the color map
            # In other words, the center of color map will be exactly zero. This
            # helps to create some consistency when showing seismic images (the
            # white color is always a zero in the seismic cmap)
            maxv[i] = max(np.abs(maxv[i]), np.abs(minv[i]))
            minv[i] = - maxv[i]
            
            # #The below sets the mean of the image at the center of the color map
            # u = Im.mean()  #Mean of the image
            # delta = max(maxv[i] - u, u - minv[i])
            # maxv[i] = u + delta
            # minv[i] = u - delta
        
    # -------------------------------------------------------------------------
    # Now set the range of ALL images for real :)
    # -------------------------------------------------------------------------
    if type(rang) == type(-1):
        # If the range was provided as an index, then use the range of I[index]
        # for all images
        a = minv[rang]
        b = maxv[rang]
        minv = a * np.ones(n)
        maxv = b * np.ones(n)
    else:
        try:
            try:
                # If rang was provided as a tuple of two points, then use
                # them for the range.
                a, b = rang
                minv = a * np.ones(n)
                maxv = b * np.ones(n)
            except:
                # If rang was provided as an image, then extract its range and
                # use it here (also, clip the range if clip>0)
                # First, convert it to numpy
                # Convert the image to a numpy array (in case it was a tensor)
                try:  #Assume the image is on the CPU and has no gradients (tensor)
                    rang = np.array(rang, dtype=float)  #To make sure the image is of type float
                except:  #If this failed, then move the image to the cpu and detach the gradients
                    rang = np.array(rang.cpu().detach(), dtype=float)  #To make sure the image is of type float
                # Then, find the range based on the provided image
                if clip != 0:
                    # The below clips the range (VERY useful for seismic images)
                    a = np.percentile(rang,     clip)  #To set the minimum value of the image
                    b = np.percentile(rang, 100-clip)  #To set the maximum value of the image
                    minv = a * np.ones(n)
                    maxv = b * np.ones(n)
                else:
                    # Find the minimum and maximum values
                    a = rang.min()  #To set the minimum value of the image
                    b = rang.max()  #To set the maximum value of the image
                    minv = a * np.ones(n)
                    maxv = b * np.ones(n)
                if rangZeroCenter:
                    # Ensuring the zero value is exactly at the center of the 
                    # color map
                    b = max(np.abs(maxv), np.abs(minv))
                    a = - b
                    minv = a * np.ones(n)
                    maxv = b * np.ones(n)
        except:
            # If the above "somehow" returned an error, then simply use minimum
            # and maximum values across ALL images
            if rang == 'norm':
                minv = minv
                maxv = maxv
            elif rang != 'global':
                error = \
'''The provided rang is not an index, a tuple of size 2, or an array!
No range can be extracted from the provided value.
As such, the global minimum and maximum values will be used'''
                warnings.warn(error, UserWarning)
            
                # Set the global minimum and maximum values
                a = minv.min()
                b = maxv.max()
                minv = a * np.ones(n)
                maxv = b * np.ones(n)
            else:
                # Set the global minimum and maximum values
                a = minv.min()
                b = maxv.max()
                minv = a * np.ones(n)
                maxv = b * np.ones(n)
    idx = -1
    
    # The below if statement can be useful when using the figures for research 
    # papers. However, there is currently a bug where the letters sometimes do
    # not show up depending on the grid shape. It works fine for grid=(1,3), 
    # but not for grid=(2,2). Finally, I could've actually messed up the order
    # of the alphabet, or better yet, forgotten the existence of some letters,
    # so please do not judge XD
    if alphabet:
        alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        xlabels = ['(' + character + ')' for character in alphabet]
        alphabet = True
    else:
        xlabels = ['' for i in range(100)]
    
    if cmap is None:
        if fft:
            # print('cmap =', cmap) 
            cmap = 'viridis'
        else:
            cmap = 'seismic_r'
        # t = [np.linspace(-0.5, 0.5, len(tim)*fft_factor)/(tim[1]-tim[0]) for tim in t]
        # offset = [np.linspace(-0.5, 0.5, len(off)*fft_factor)/(off[1]-off[0]) for off in offset]
    
    for ax in grid:
        idx += 1
        # The below 
        try:  #Assume the image is on the CPU and has no gradients
            Im = np.array(I[idx], dtype=float)  #To make sure the image is of type float
        except:
            try: #If the above failed, then move the image to the cpu and 
                 #detach the gradients
                Im = np.array(I[idx].cpu().detach(), dtype=float)  #To make sure the image is of type float
            except:
                fig.delaxes(ax)
                continue  #Skip to the next image
        # If the image was blank, then remove it from the figure
        if Im.std() == 0.00 and ignoreZeroStd:
            fig.delaxes(ax)
            continue  #Skip to the next image
        # plt.subplot(grid[0],grid[1], subplot[i])
        if alphabet:
            ax.set_xlabel(xlabels[idx])  #Buggy. Does not show the labels sometimes
            # print('Printing the alphabetical order of the subplots...')
        if disable_Axes:
            ax.axis('off')
            ax.tick_params(left=False,bottom=False)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            if fft:
                ax.set_xlabel(f'Wavenumber (1/m)\n{xlabels[idx]}', fontsize=fontsize)
                ax.set_ylabel('Frequency (Hz)', fontsize=fontsize)
            else:
                # ax.set_xlabel(f'Offset (km)\n{xlabels[idx]}', fontsize=fontsize)
                ax.set_xlabel(f'Trace\n{xlabels[idx]}', fontsize=fontsize)
                ax.set_ylabel('Time (s)', fontsize=fontsize)
        ax.set_title(title[idx], fontsize=fontsize)
        # try:
        #     plt.title(title[i])
        # except:
        #     print('The number of titles is less than the number of image!')
        
        # n = int(np.ceil(np.log2(1+np.amax(np.array(Im))
        #                         - np.amin(np.array(Im)))))  #Number of bits required to represent x
        # # n = int(np.ceil(np.log2(np.amax(np.array(Im))+1)))  #Number of bits required to represent x
        # L = 2**n  #Number of intensity levels available in the image (in other words, it is our bandwidth)
        # minimum = np.amin(Im)
        
        # =====================================================================
        # Fix the dimensions of the image
        # =====================================================================
        #If the image has the channels in the first axis, then move it to the last
        if len(Im.shape) == 3 and min(Im.shape) == Im.shape[0]:
            # print('Moved the channels axis to align with plt.imshow')
            Im = np.moveaxis(Im, 0, 2)
        
        if len(Im.shape) == 4:  #If the image was a tensor (most likely)
                # Use the first sample to convert it to a 3D array, and then
                # move the channel axis to the last dimension
                Im = np.moveaxis(Im[0],0,-1)  
        
        # The below block is used for the extent
        t_min = t[idx].min().item()
        t_max = t[idx].max().item()
        offset_min = offset[idx].min().item()
        offset_max = offset[idx].max().item()
        if disable_Axes:
            square_Axes = 1
        else:
            # print(t)
            square_Axes = np.ptp(offset[idx]).item() / np.ptp(t[idx]).item()
        # print(f'Disable axes is {disable_Axes}')
        if len(Im.shape) == 2 or (len(Im.shape) == 3 and Im.shape[2] == 1): #If the image was grayscale,
            
            show = ax.imshow(Im, cmap=cmap, vmin=minv[idx], vmax=maxv[idx], 
                             aspect=aspect*square_Axes, 
                             extent=[offset_min, offset_max, t_max, t_min],
                             **kwargs)
            # if L == 256 and (minimum >= 0):
            #     plt.imshow(np.uint8(Im), cmap=cmap, vmin=0, vmax=255)
            # elif L==1 and (minimum >= 0):
            #     plt.imshow(np.uint8(Im*255), cmap=cmap, vmin=0, vmax=255)
            # else:
                # plt.imshow(normIm(Im, 0, 255), cmap=cmap, vmin=0, vmax=255)
                # print("The image does NOT have a standard range... \nNormalizing...")
        elif len(Im.shape) == 3:
            if Im.shape[-1] == 1: #If the image was grayscale,
                show = ax.imshow(Im, cmap=cmap, vmin=minv[idx], vmax=maxv[idx], 
                                 aspect=aspect*square_Axes, 
                                 extent=[offset_min, offset_max, t_max, t_min],
                                 **kwargs)
                # if L == 256 and (minimum >= 0):
                #     plt.imshow(np.uint8(Im), cmap=cmap, vmin=0, vmax=255)
                # elif L==1 and (minimum >= 0):
                #     plt.imshow(np.uint8(Im*255), cmap=cmap, vmin=0, vmax=255)
                # else:
                #     print("The image does NOT have a standard range... \nNormalizing...")
                #     plt.imshow(normIm(Im, 0, 255), cmap=cmap, vmin=0, vmax=255)
            elif Im.shape[-1] == 3: #If the image was RGB,
                # print(f'vmin={Im.min()}, vmax={Im.max()}')
                show = ax.imshow(Im, vmin=minv[idx], vmax=maxv[idx], 
                                 aspect=aspect*square_Axes, 
                                 # extent=[offset_min, offset_max, t_max, t_min],
                                 **kwargs)
                # if L == 256 and (minimum >= 0):
                #     plt.imshow(np.uint8(Im), vmin=0, vmax=255)
                # elif L==1 and (minimum >= 0):
                #     plt.imshow(np.uint8(Im*255), vmin=0, vmax=255)
                # else:
                #     print("The image does NOT have a standard range... \nNormalizing...")
                #     plt.imshow(normIm(Im, 0, 255), vmin=0, vmax=255)
            else:
                # Show only the first channel
                show = ax.imshow(Im[:,:,0], cmap=cmap, vmin=minv[idx], vmax=maxv[idx], 
                                 aspect=aspect*square_Axes, 
                                 extent=[offset_min, offset_max, t_max, t_min],
                                 **kwargs)
        else:
            print("Something isn't right with the dimensions of the image, so fix it!")
            #The above line is problematic
    if colorbar:
        # from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
        # aspect = 20
        # pad_fraction = 0.5
        # divider = make_axes_locatable(ax)
        # width = axes_size.AxesY(ax, aspect=1./aspect)
        # pad = axes_size.Fraction(pad_fraction, width)
        # cax = divider.append_axes("right", size=width, pad=pad)
        # cax = divider.append_axes("right", size="50%", pad="10%")
        # cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
        # fig.colorbar(show, cax=cax) # Similar to fig.colorbar(im, cax = cax)
        # fig.colorbar(show, ax=ax.ravel().tolist()) # Similar to fig.colorbar(im, cax = cax)
        # fig.colorbar(show, ax=ax) # Similar to fig.colorbar(im, cax = cax)
        
        # Colorbar
        # colormap = plt.cm.get_cmap(cmap)
        # sm = plt.cm.ScalarMappable(cmap=colormap)
        # sm.set_clim(vmin=minv, vmax=maxv)
        # plt.colorbar(sm)
        # cbar = ax.cax.colorbar(sm)
        
        # Temporarily change the rcParam to solve deprecation warning
        with mpl.rc_context({'mpl_toolkits.legacy_colorbar': False}):
            # consider cbar = fig.colorbar('something here')
            cbar = ax.cax.colorbar(show)
            cbar.ax.locator_params(nbins=cbar_ticks)
            if cbar_label != None:
                cbar.ax.set_ylabel(cbar_label, fontsize=fontsize) #, rotation=270)
        
        #The below method seems more "official"
        # ax.cax.colorbar(show)
        # ax.cax.toggle_label(True)
    # fig.tight_layout(pad=5)


'''
Links that might prove to be useful in the future:
    - ImageGrid with varying extents fails to maintain padding (https://github.com/matplotlib/matplotlib/issues/8695)
    - Foregoing the use of ImageGrid https://stackoverflow.com/questions/66292311/how-to-change-the-height-of-each-image-grid-with-mpl-toolkits-axes-grid1-imagegr
'''
