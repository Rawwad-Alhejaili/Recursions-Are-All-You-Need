#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is simply a wrapper for imshow2_offset.py. The goal behind it is to store
your defaults only once. For example, I need to constantly show a grid of 2x2
with no color bar, and also have to show a grid of 1x4 that DO include a color
bar. Instead of having to change the arguments for each case, I can now define 
2 imshow instances with different defaults so I don't have to constantly change 
the arguments back and forth (which can still be temporarily overwritten even 
after setting the new defaults).

@author: Someone lazy :)
"""

from copy import deepcopy  #to copy the image regardless of whether its numpy or tensor
from imshow2_offset import imshow2

class lazy_imshow():
    def __init__(self, 
                 I              = None, 
                 title          = None, 
                 grid           = (1,1), 
                 colorbar       = True, 
                 cbar_ticks     = 11, 
                 cbar_label     = None,
                 aspect         = 1,
                 cmap           = 'seismic_r', 
                 pad            = 0.7, 
                 plotsize       = (7,7), 
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
                 **kwargs):
        #Store the defaults in self
        self.I              = I
        self.title          = title
        self.grid           = grid
        self.colorbar       = colorbar
        self.cbar_ticks     = cbar_ticks
        self.cbar_label     = cbar_label
        self.aspect         = aspect
        self.cmap           = cmap
        self.pad            = pad
        self.plotsize       = plotsize
        self.rang           = rang
        self.rangZeroCenter = rangZeroCenter
        self.dpi            = dpi
        self.figTransparent = figTransparent
        self.fontsize       = fontsize
        self.alphabet       = alphabet
        self.ignoreZeroStd  = ignoreZeroStd
        self.clip           = clip
        self.t              = t
        self.offset         = offset
        self.fft            = fft
        
    def imshow(self, 
             I              = None,
             title          = None, 
             grid           = None, 
             colorbar       = None, 
             cbar_ticks     = None, 
             cbar_label     = None,
             aspect         = None,
             cmap           = None, 
             pad            = None, 
             plotsize       = None, 
             rang           = None, 
             rangZeroCenter = None, 
             dpi            = None, 
             figTransparent = None, 
             fontsize       = None, 
             alphabet       = None, 
             ignoreZeroStd  = None, 
             clip           = None,
             t              = None,
             offset         = None,
             fft            = None,
             **kwargs):
        
        # If the arguments are not changed, use the defaults from __init__
        if I is None:
            I     = deepcopy(self.I)
        if title          is None:
            title = deepcopy(self.title)
        if grid           == None:
            grid           = self.grid
        if colorbar       == None:
            colorbar       = self.colorbar
        if cbar_ticks     == None:
            cbar_ticks     = self.cbar_ticks
        if cbar_label     == None:
            cbar_label     = self.cbar_label
        if aspect         == None:
            aspect         = self.aspect
        if cmap           == None:
            cmap           = self.cmap
        if pad            == None:
            pad            = self.pad
        if plotsize       == None:
            plotsize       = self.plotsize
        if rang           == None:
            rang           = self.rang
        if rangZeroCenter == None:
            rangZeroCenter = self.rangZeroCenter
        if dpi            == None:
            dpi            = self.dpi
        if figTransparent == None:
            figTransparent = self.figTransparent
        if fontsize       == None:
            fontsize       = self.fontsize
        if alphabet       == None:
            alphabet       = self.alphabet
        if ignoreZeroStd  == None:
            ignoreZeroStd  = self.ignoreZeroStd
        if clip           == None:
            clip           = self.clip
        if t              is None:
            t              = self.t
        if offset         is None:
            offset         = self.offset
        if fft            == None:
            fft            = self.fft
        
        # Show the image
        return \
        imshow2(I,
             title,
             grid,
             colorbar,
             cbar_ticks,
             cbar_label,
             aspect,
             cmap,
             pad,
             plotsize,
             rang,
             rangZeroCenter,
             dpi,
             figTransparent,
             fontsize,
             alphabet,
             ignoreZeroStd,
             clip,
             t,
             offset,
             fft,
             **kwargs)
        
        # Note:
        # When calling imshowSeismic2, I could have used "named" arguments to
        # to avoid errors in the future (if I update the locations of 
        # imshowSeismic2's arguments, then errors will be raised). However, I
        # have intentionally did it this way to catch errors early on.

# # =============================================================================
# # Initialization Example (Outdated)
# # =============================================================================
# lazy = lazy_imshow(I              = None, 
#                    title          = '', 
#                    grid           = (1,1), 
#                    colorbar       = True, 
#                    cbar_ticks     = 11,
#                    aspect         = 1,
#                    cmap           = 'seismic_r', 
#                    pad            = 0.7, 
#                    plotsize       = (10,10), 
#                    rang           = -1, 
#                    rangZeroCenter = False, 
#                    dpi            = 300, 
#                    figTransparent = False, 
#                    fontsize       = 14, 
#                    alphabet       = False,
#                    ignoreZeroStd  = True, 
#                    clip           = 0)

# imshow = lazy.imshow
# # Now call imshow(yourImages) to display the images with your default
# # parameters. Note that you can temporarily override the default parameters.
# # For example, assume we have a grid of (1,4), and we would like to change it
# # to (2,2), then call:
    
# # imshow(yourImage, grid=(2,2))

# # Now the image will be displayed your chosen grid. 
# # If you wish to permenantly change the default parameters, then you will have 
# # to call lazyImshow again.
