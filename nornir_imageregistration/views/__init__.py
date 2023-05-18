__all__ = ['transformwarp', 'alignment_records', 'display_images', 'TransformWarpView', 'StosTransformWarpView',
           'ShowGrayscale', 'plot_tile_overlaps', 'plot_tile_overlap', 'plot_layout', 'plot_aligned_images']

import matplotlib
import matplotlib.pyplot as plt

from nornir_imageregistration.views.alignment_records import PlotWeightHistogram, PlotPeakList, plot_aligned_images
from nornir_imageregistration.views.display_images import ShowGrayscale
from nornir_imageregistration.views.layout import plot_layout
from nornir_imageregistration.views.tile_overlap import plot_tile_overlap, plot_tile_overlaps
from nornir_imageregistration.views.transformwarp import TransformWarpView, StosTransformWarpView


class PassFailInput(object):

    def __init__(self, fig):
        self.Pass = None
        self.fig = fig

    def OnPassButton(self, event):
        self.Pass = True
        return

    def OnFailButton(self, event):
        self.Pass = False
        return


def ShowWithPassFail(fig):
    '''Shows the prepared figure with the addition of two "Pass/Fail" buttons
       return True if the pass button is pressed.  Otherwise false
    '''
    callback = PassFailInput(fig)
    axprev = plt.axes([0.7, 0.05, 0.1, 0.075])
    axnext = plt.axes([0.81, 0.05, 0.1, 0.075])

    bnext = matplotlib.widgets.Button(axnext, 'Pass', color='#00FF80')
    bnext.on_clicked(callback.OnPassButton)
    bprev = matplotlib.widgets.Button(axprev, 'Fail', color='#FF0000')
    bprev.on_clicked(callback.OnFailButton)
    fig.show()

    try:
        while callback.Pass is None:
            fig.waitforbuttonpress(timeout=1)
    finally:
        plt.close(fig)

    return callback.Pass
