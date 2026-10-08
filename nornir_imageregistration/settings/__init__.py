import json
import logging

from .grid_refinement import GridRefinement
from .mosaic_tile_offset import LoadMosaicOffsets, SaveMosaicOffsets, TileOffset
from .translate import TranslateSettings
from .angle_range import AngleSearchRange
from .stos_brute import StosBruteSettings, SliceToSliceMethod

logger = logging.getLogger(__name__)

_RECOVERABLE_READ_ERRORS = (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError)


def _write_translate_settings(settings: TranslateSettings, path: str) -> TranslateSettings:
    with open(path, 'w', encoding='utf-8') as jsonfile:
        json.dump(settings.__dict__, jsonfile, sort_keys=True, indent=2)
    return settings


def GetOrSaveTranslateSettings(settings: TranslateSettings | None, path: str) -> TranslateSettings:
    '''
    Check if a .json file exists, if it does load and return it.  Otherwise
    save the provide settings file as a .json file
    '''
    try:
        with open(path, encoding='utf-8') as jsonfile:
            data = json.load(jsonfile)
        if not isinstance(data, dict):
            raise TypeError(f"Translate settings JSON root must be an object, got {type(data).__name__}")
        return TranslateSettings(**data)
    except FileNotFoundError:
        if settings is None:
            raise
        logger.info("Creating translate settings at %s", path)
        return _write_translate_settings(settings, path)
    except _RECOVERABLE_READ_ERRORS as exc:
        if settings is None:
            raise
        logger.warning(
            "Could not load translate settings from %s (%s: %s); overwriting with provided defaults",
            path,
            type(exc).__name__,
            exc,
        )
        return _write_translate_settings(settings, path)
