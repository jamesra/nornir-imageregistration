"""``figure_tag`` derives an artifact filename tag from what a figure actually says.

Headless plot artifacts are triaged after the run, which means each PNG has to be traceable
back to the code that drew it.  Callers rarely pass an explicit tag, so ``ShowWithPassFail``
used to name every artifact ``passfail-<pid>-<uuid>.png`` -- one identical prefix for every
figure in the session, with a uuid as the only distinguishing part.

The tag is filename material, so the two properties that matter are that it stays
filesystem-safe and bounded in length, and that it never raises: a figure that cannot be
interrogated must degrade to the default rather than take down the test that drew it.
"""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from nornir_imageregistration.headless import _TAG_MAX_LENGTH, figure_tag


class TestItPrefersTheSuptitle(unittest.TestCase):

    def setUp(self):
        self.fig = plt.figure()
        self.addCleanup(plt.close, self.fig)

    def test_a_suptitle_becomes_a_slug(self):
        self.fig.suptitle('Refine pass 3: cell 12 rejected')

        self.assertEqual('refine-pass-3-cell-12-rejected', figure_tag(self.fig))

    def test_the_suptitle_wins_over_an_axes_title(self):
        self.fig.suptitle('outer')
        self.fig.add_subplot(1, 1, 1).set_title('inner')

        self.assertEqual('outer', figure_tag(self.fig))

    def test_an_axes_title_is_used_when_there_is_no_suptitle(self):
        self.fig.add_subplot(1, 1, 1).set_title('Target vs source')

        self.assertEqual('target-vs-source', figure_tag(self.fig))

    def test_the_first_titled_axes_wins(self):
        self.fig.add_subplot(2, 1, 1).set_title('first')
        self.fig.add_subplot(2, 1, 2).set_title('second')

        self.assertEqual('first', figure_tag(self.fig))

    def test_an_untitled_axes_is_skipped_rather_than_returning_an_empty_tag(self):
        self.fig.add_subplot(2, 1, 1)
        self.fig.add_subplot(2, 1, 2).set_title('second')

        self.assertEqual('second', figure_tag(self.fig))


class TestTheTagIsSafeToPutInAFilename(unittest.TestCase):

    def setUp(self):
        self.fig = plt.figure()
        self.addCleanup(plt.close, self.fig)

    def test_path_separators_and_punctuation_do_not_survive(self):
        self.fig.suptitle(r'a/b\c:d*e?f"g<h>i|j')

        tag = figure_tag(self.fig)

        for forbidden in r'/\:*?"<>|':
            self.assertNotIn(forbidden, tag)

    def test_it_is_truncated_and_does_not_end_in_a_separator(self):
        self.fig.suptitle('word ' * 40)

        tag = figure_tag(self.fig)

        self.assertLessEqual(len(tag), _TAG_MAX_LENGTH)
        self.assertFalse(tag.endswith('-'))

    def test_leading_and_trailing_punctuation_is_stripped(self):
        self.fig.suptitle('  ...leading and trailing...  ')

        tag = figure_tag(self.fig)

        self.assertFalse(tag.startswith('-'))
        self.assertFalse(tag.endswith('-'))


class TestItFallsBackRatherThanRaising(unittest.TestCase):
    """A tag is a filename convenience; failing to derive one must not fail the caller."""

    def test_a_figure_with_no_text_uses_the_default(self):
        fig = plt.figure()
        self.addCleanup(plt.close, fig)

        self.assertEqual('fig', figure_tag(fig))
        self.assertEqual('untitled', figure_tag(fig, default='untitled'))

    def test_a_title_with_no_usable_characters_uses_the_default(self):
        fig = plt.figure()
        self.addCleanup(plt.close, fig)
        fig.suptitle('***')

        self.assertEqual('untitled', figure_tag(fig, default='untitled'))

    def test_an_object_that_is_not_a_figure_uses_the_default(self):
        self.assertEqual('fig', figure_tag(object()))
        self.assertEqual('fig', figure_tag(None))

    def test_a_figure_whose_accessors_raise_uses_the_default(self):
        class Hostile:
            @property
            def axes(self):
                raise RuntimeError('no axes for you')

            def get_suptitle(self):
                raise RuntimeError('no suptitle either')

        self.assertEqual('fallback', figure_tag(Hostile(), default='fallback'))


if __name__ == '__main__':
    unittest.main()
