__all__ = ['transformwarp', 'alignment_records', 'display_images', 'TransformWarpView', 'StosTransformWarpView',
           'ShowGrayscale', 'plot_tile_overlaps', 'plot_tile_overlap', 'plot_layout', 'plot_aligned_images',
           'ShowWithPassFail', 'plot_percentiles']

import matplotlib
import matplotlib.pyplot as plt

from nornir_imageregistration.headless import is_headless, save_figure_to_png_artifact

from nornir_imageregistration.views import transformwarp, alignment_records, display_images
from nornir_imageregistration.views.alignment_records import PlotPeakList, PlotWeightHistogram, plot_aligned_images, \
    plot_percentiles
from nornir_imageregistration.views.grid_refinement import ShowGridRefinement, build_grid_rois
from nornir_imageregistration.views.grid_data import PlotGridPositionsAndMask
from nornir_imageregistration.views.display_images import ShowGrayscale
from nornir_imageregistration.views.layout import plot_layout
from nornir_imageregistration.views.tile_overlap import plot_tile_overlap, plot_tile_overlaps
from nornir_imageregistration.views.transformwarp import StosTransformWarpView, TransformWarpView


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
    if is_headless():
        save_figure_to_png_artifact(fig, tag="passfail", dpi=150)
        return True

    callback = PassFailInput(fig)
    axprev = plt.axes((0.7, 0.05, 0.1, 0.075))
    axnext = plt.axes((0.81, 0.05, 0.1, 0.075))

    bnext = matplotlib.widgets.Button(axnext, 'Pass', color='#00FF80')  # type: ignore[attr-defined]
    bnext.on_clicked(callback.OnPassButton)
    bprev = matplotlib.widgets.Button(axprev, 'Fail', color='#FF0000')  # type: ignore[attr-defined]
    bprev.on_clicked(callback.OnFailButton)
    fig.show()

    try:
        while callback.Pass is None:
            fig.waitforbuttonpress(timeout=1)
    finally:
        plt.close(fig)

    return callback.Pass
