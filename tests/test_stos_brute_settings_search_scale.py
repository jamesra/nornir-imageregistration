"""``search_scale`` must exist on ``StosBruteSettings``, not just in the callers' signatures.

``SliceToSliceRigidRegistration`` grew a ``search_scale`` parameter and forwards it into
``StosBruteSettings(...)``, and four sites in ``stos_brute`` read ``settings.search_scale``,
but the field was never added to the settings model. ``StosBruteSettings`` is a pydantic
``BaseModel`` with an explicit ``__init__`` that takes no ``**kwargs``, so the keyword was a
hard ``TypeError`` and the public entry point failed on **every** call, with all arguments
defaulted (#258).

Two halves have to be present, which is why this asserts both:

* the ``__init__`` keyword, or construction raises ``TypeError``; and
* the model *field*, or the four ``settings.search_scale`` reads raise ``AttributeError``.

Fixing only the first leaves the second, and a test that merely constructed the settings
would have passed.

The default must stay ``True``: every existing caller relies on the scale probe, search and
refinement running, and ``False`` deliberately reduces the registration to a translation
(plus angle range) match at scale 1.0.
"""

from __future__ import annotations

import inspect
import unittest

from nornir_imageregistration import stos_brute
from nornir_imageregistration.settings.stos_brute import (
    SliceToSliceMethod,
    StosBruteSettings,
)


class TestSearchScaleIsPartOfTheSettings(unittest.TestCase):

    def test_the_field_exists_on_the_model(self):
        """The half that the four settings.search_scale reads depend on."""
        self.assertIn('search_scale', StosBruteSettings.model_fields)

    def test_the_constructor_accepts_the_keyword(self):
        """The half that the public entry point depends on."""
        self.assertIn('search_scale', inspect.signature(StosBruteSettings.__init__).parameters)

    def test_it_defaults_to_true(self):
        self.assertTrue(StosBruteSettings().search_scale)
        self.assertTrue(StosBruteSettings(method=SliceToSliceMethod.BruteForce).search_scale)

    def test_it_round_trips_both_ways(self):
        for requested in (True, False):
            with self.subTest(search_scale=requested):
                settings = StosBruteSettings(search_scale=requested)
                self.assertEqual(requested, settings.search_scale)

    def test_setting_it_does_not_disturb_the_other_fields(self):
        settings = StosBruteSettings(method=SliceToSliceMethod.BruteForce,
                                     min_overlap=0.25,
                                     larget_dimension=512,
                                     try_flipped=False,
                                     initial_scale_hint=0.98,
                                     search_scale=False)

        self.assertFalse(settings.search_scale)
        self.assertEqual(SliceToSliceMethod.BruteForce, settings.method)
        self.assertEqual(0.25, settings.min_overlap)
        self.assertEqual(512, settings.larget_dimension)
        self.assertFalse(settings.try_flipped)
        self.assertEqual(0.98, settings.initial_scale_hint)


class TestThePublicEntryPointCanBuildItsSettings(unittest.TestCase):
    """The observable failure: the TypeError fired before any registration work began."""

    def test_the_entry_point_declares_search_scale(self):
        parameters = inspect.signature(stos_brute.SliceToSliceRigidRegistration).parameters

        self.assertIn('search_scale', parameters)
        self.assertIs(True, parameters['search_scale'].default,
                      'flipping this default would change every existing registration')

    def test_the_settings_the_entry_point_builds_can_be_constructed(self):
        """Mirrors the StosBruteSettings(...) call at stos_brute.py:944 argument for argument."""
        for search_scale in (True, False):
            with self.subTest(search_scale=search_scale):
                settings = StosBruteSettings(method=SliceToSliceMethod.LogPolar,
                                             angles={0.0, 2.0},
                                             min_overlap=0.5,
                                             source_image_scale_factors=None,
                                             larget_dimension=1024,
                                             try_flipped=True,
                                             estimated_scale_hint=None,
                                             initial_scale_hint=None,
                                             search_scale=search_scale)

                self.assertEqual(search_scale, settings.search_scale)


if __name__ == '__main__':
    unittest.main()
