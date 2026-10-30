"""Tests for nested dashboard progress on image_stats.Histogram."""

from __future__ import annotations

import unittest
from unittest import mock

from nornir_shared.histogram import Histogram


class TestHistogramProgress(unittest.TestCase):
    def test_histogram_publishes_task_progress_and_complete(self) -> None:
        from nornir_imageregistration import image_stats

        fake_hist = Histogram.Init(minVal=0, maxVal=255, numBins=256)
        paths = ["a.png", "b.png"]

        with mock.patch.object(image_stats, "__HistogramFileSciPy__", return_value=fake_hist):
            with mock.patch("nornir_shared.prettyoutput.publish_task_progress") as progress:
                with mock.patch("nornir_shared.prettyoutput.publish_task_complete") as complete:
                    result = image_stats.Histogram(
                        paths,
                        Bpp=8,
                        progress_task_key="import_idoc:histogram",
                        progress_name="Import histogram 1",
                    )

        self.assertIsNotNone(result)
        self.assertGreaterEqual(progress.call_count, 2)

        # Compare on (positional args, name) only. publish_task_progress also takes
        # optional dashboard metadata (element/path/section) that Histogram passes as
        # None; matching the full kwarg set would break every time one is added.
        published = [(call.args, call.kwargs.get("name")) for call in progress.call_args_list]
        self.assertIn((("import_idoc:histogram", 0, 2), "Import histogram 1"), published)
        self.assertIn((("import_idoc:histogram", 2, 2), "Import histogram 1"), published)
        complete.assert_called_once_with("import_idoc:histogram", 2)

    def test_histogram_skips_progress_without_task_key(self) -> None:
        from nornir_imageregistration import image_stats

        fake_hist = Histogram.Init(minVal=0, maxVal=255, numBins=256)

        with mock.patch.object(image_stats, "__HistogramFileSciPy__", return_value=fake_hist):
            with mock.patch("nornir_shared.prettyoutput.publish_task_progress") as progress:
                with mock.patch("nornir_shared.prettyoutput.publish_task_complete") as complete:
                    image_stats.Histogram(["a.png"], Bpp=8)

        progress.assert_not_called()
        complete.assert_not_called()


if __name__ == "__main__":
    unittest.main()
