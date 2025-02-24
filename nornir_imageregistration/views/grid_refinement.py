import nornir_imageregistration
from nornir_imageregistration.settings import GridRefinement
import nornir_imageregistration.settings as settings
from nornir_imageregistration.grid_subdivision import IGrid


def build_grid_rois(grid: IGrid):
    """Build a list of rois for a grid"""
    source_rois = []
    target_rois = []

    for point in grid.SourcePoints:
        source_rect = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(point, grid.cell_size)
        source_rois.append(source_rect)

    for point in grid.TargetPoints:
        target_rect = nornir_imageregistration.Rectangle.CreateFromCenterPointAndArea(point, grid.cell_size)
        target_rois.append(target_rect)

    return source_rois, target_rois


def ShowGridRefinement(settings: settings.GridRefinement, grid: IGrid, passfail: bool = False,
                       filename: str | None = None):
    source_rois, target_rois = build_grid_rois(grid)

    nornir_imageregistration.ShowGrayscale(
        input_params=[settings.source_image, settings.source_mask, settings.target_image, settings.target_mask],
        title="Masked Grid, Source and Target Images with grid cells overlayed, target rectangles are not rotated",
        image_titles=["Source", "Source Mask", "Target", "Target Mask"],
        rois=[source_rois, source_rois, target_rois, target_rois],
        PassFail=passfail,
        filename=filename)
    return
