"""Transient share errors (EBUSY/EAGAIN) on raw reads are retried instead of aborting an import."""
import errno
import os
import tempfile

import numpy as np
import pytest
from nornir_imageregistration import image_stats, io_retry, noise_crush
from PIL import Image


def _flaky(failures: int, error_number: int, value=42):
    calls = {'n': 0}

    def read():
        calls['n'] += 1
        if calls['n'] <= failures:
            raise OSError(error_number, os.strerror(error_number))
        return value

    return read, calls


class TestCallWithIoRetry:
    def test_retries_ebusy_then_succeeds(self):
        read, calls = _flaky(2, errno.EBUSY)
        delays = []
        assert io_retry.call_with_io_retry(read, attempts=5, base_delay=0.5, sleep=delays.append) == 42
        assert calls['n'] == 3
        assert delays == [0.5, 1.0]

    def test_retries_eagain(self):
        read, calls = _flaky(1, errno.EAGAIN)
        assert io_retry.call_with_io_retry(read, attempts=5, base_delay=0.0, sleep=lambda s: None) == 42
        assert calls['n'] == 2

    def test_gives_up_after_attempts(self):
        read, calls = _flaky(10, errno.EBUSY)
        with pytest.raises(OSError) as info:
            io_retry.call_with_io_retry(read, attempts=5, base_delay=0.0, sleep=lambda s: None)
        assert info.value.errno == errno.EBUSY
        assert calls['n'] == 5

    def test_other_errors_are_not_retried(self):
        read, calls = _flaky(1, errno.ENOENT)
        with pytest.raises(OSError):
            io_retry.call_with_io_retry(read, attempts=5, base_delay=0.0, sleep=lambda s: None)
        assert calls['n'] == 1


@pytest.fixture
def raw_tif():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, 'tile.tif')
        Image.fromarray(np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)).save(path)
        yield path


def _ebusy_then_real(monkeypatch, module, failures: int):
    real_open = Image.open
    calls = {'n': 0}

    def fake_open(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] <= failures:
            raise OSError(errno.EBUSY, 'Device or resource busy')
        return real_open(*args, **kwargs)

    monkeypatch.setattr(module.Image, 'open', fake_open)
    monkeypatch.setattr(noise_crush, 'RAW_READ_BASE_DELAY', 0.0)
    return calls


def test_load_raw_tile_survives_transient_ebusy(monkeypatch, raw_tif):
    calls = _ebusy_then_real(monkeypatch, noise_crush, failures=2)
    tile = noise_crush.load_raw_tile(raw_tif)
    assert calls['n'] == 3
    assert tile.dtype == np.uint16 and tile[1, 0] == 64


def test_load_raw_tile_still_fails_on_persistent_ebusy(monkeypatch, raw_tif):
    _ebusy_then_real(monkeypatch, noise_crush, failures=100)
    with pytest.raises(OSError):
        noise_crush.load_raw_tile(raw_tif)


def test_histogram_read_survives_transient_ebusy(monkeypatch, raw_tif):
    calls = _ebusy_then_real(monkeypatch, image_stats, failures=1)
    histogram = image_stats.Histogram([raw_tif], Bpp=16)
    assert calls['n'] >= 2
    assert sum(histogram.Bins) == 64 * 64


def test_load_image_survives_connection_abort(monkeypatch, tmp_path):
    """LoadImage rides out ECONNABORTED while a CIFS share reconnects (RC3, storage2, 2026-11-21)."""
    import nornir_imageregistration
    import nornir_shared.files
    from nornir_imageregistration.core import _core

    path = str(tmp_path / '889.png')
    Image.fromarray(np.full((8, 8), 7, dtype=np.uint8)).save(path)
    real_once = _core._load_image_by_extension_once
    calls = {'n': 0}

    def flaky(*args, **kwargs):
        calls['n'] += 1
        if calls['n'] == 1:
            raise OSError(errno.ECONNABORTED, os.strerror(errno.ECONNABORTED))
        return real_once(*args, **kwargs)

    monkeypatch.setattr(_core, '_load_image_by_extension_once', flaky)
    monkeypatch.setattr(nornir_shared.files.time, 'sleep', lambda _s: None)
    image = nornir_imageregistration.LoadImage(path, dtype=np.uint8)
    assert calls['n'] == 2
    assert int(nornir_imageregistration.EnsureNumpyArray(image).max()) == 7
