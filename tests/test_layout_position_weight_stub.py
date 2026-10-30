"""`ScaleOffsetWeightsByPosition` must fail with its own message, not an arithmetic one.

The function has never been implemented -- it has always ended in an unconditional
``raise NotImplementedError``. That raise used to sit *after* a weight computation, so what a
caller actually saw depended on the layout: under this module's numpy error state
(``divide='raise'``, ``invalid='raise'``) a layout whose offsets agree with its positions has
zero tension everywhere, making the median distance zero and ``distance / medianDistance``
a 0/0 that surfaced as ``FloatingPointError: invalid value encountered in divide``.

That reads like an arithmetic bug in a working function rather than an unfinished one. These
tests pin the honest failure, and pin that the three sibling helpers repaired in #130 still work.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

import nornir_imageregistration.layout as layout_module
from nornir_imageregistration.layout import Layout, LayoutPosition


def _layout(offsets: list[tuple[float, float]]) -> Layout:
    """Build a chain of collinear nodes 10 apart, linked by the given (offset, weight) pairs."""
    layout = Layout()
    for i in range(len(offsets) + 1):
        layout.CreateNode(i, np.asarray((0.0, i * 10.0)))
    for i, (offset, weight) in enumerate(offsets):
        layout.SetOffset(i, i + 1, np.asarray((0.0, offset)), weight)
    return layout


# The offsets match the node spacing exactly, so every tension vector is zero. This is the
# case that used to raise FloatingPointError instead of NotImplementedError.
_ZERO_TENSION = [(10.0, 0.3), (10.0, 0.7)]
_NONZERO_TENSION = [(13.0, 0.3), (7.0, 0.7)]
_SINGLE_LINK = [(13.0, 0.5)]


class TestThePremise:
    """Facts the fix depends on."""

    def test_the_module_makes_bad_arithmetic_raise(self):
        err = np.geterr()
        assert err['invalid'] == 'raise'
        assert err['divide'] == 'raise'

    def test_a_zero_tension_layout_really_has_zero_tension(self):
        layout = _layout(_ZERO_TENSION)
        for node in layout.nodes.values():
            vectors = node.TensionVectors(layout.GetNodes(node.ConnectedIDs))
            assert np.allclose(vectors, 0.0)

    def test_the_only_call_site_is_commented_out(self):
        import nornir_imageregistration.arrange_mosaic as arrange
        source = inspect.getsource(arrange)
        for line in source.splitlines():
            if 'ScaleOffsetWeightsByPosition' in line:
                assert line.lstrip().startswith('#'), f'live call site: {line.strip()}'


class TestTheFailureIsHonest:

    @pytest.mark.parametrize('offsets, label', [
        (_ZERO_TENSION, 'zero tension'),
        (_NONZERO_TENSION, 'nonzero tension'),
        (_SINGLE_LINK, 'single link'),
    ])
    def test_it_raises_not_implemented_for_every_layout(self, offsets, label):
        with pytest.raises(NotImplementedError):
            layout_module.ScaleOffsetWeightsByPosition(_layout(offsets))

    def test_the_zero_tension_layout_no_longer_raises_a_float_error(self):
        """The specific regression: this used to be FloatingPointError."""
        with pytest.raises(NotImplementedError):
            layout_module.ScaleOffsetWeightsByPosition(_layout(_ZERO_TENSION))

    def test_the_message_names_the_function_and_the_reason(self):
        with pytest.raises(NotImplementedError) as excinfo:
            layout_module.ScaleOffsetWeightsByPosition(_layout(_ZERO_TENSION))
        message = str(excinfo.value)
        assert 'ScaleOffsetWeightsByPosition' in message
        assert 'implement' in message.lower()

    def test_the_method_raises_before_any_arithmetic(self):
        """Reaching the raise must not depend on the input, so nothing may precede it."""
        source = inspect.getsource(LayoutPosition.ScaleOffsetWeightsByPosition)
        body = [line.strip() for line in source.splitlines()
                if line.strip() and not line.strip().startswith('#')]
        # Drop the signature and the docstring block.
        assert body[0].startswith('def ')
        body = body[1:]
        assert body[0].startswith('"""')
        closing = next(i for i, line in enumerate(body[1:], start=1)
                       if line.endswith('"""'))
        statements = body[closing + 1:]
        assert statements, 'no statements found after the docstring'
        assert statements[0].startswith('raise NotImplementedError'), (
            f'first statement is {statements[0]!r}, not the raise')

    def test_the_unreachable_computation_is_gone(self):
        source = inspect.getsource(LayoutPosition.ScaleOffsetWeightsByPosition)
        code = '\n'.join(line for line in source.splitlines()
                         if not line.strip().startswith('#'))
        for dead in ('distance / medianDistance', 'np.median', 'TensionVectors'):
            assert dead not in code, f'{dead!r} is still present after the raise'

    def test_an_empty_layout_does_not_raise(self):
        """The wrapper iterates nodes, so with none there is nothing to delegate to."""
        layout_module.ScaleOffsetWeightsByPosition(Layout())


class TestTheSiblingsStillWork:
    """#130 repaired these three; this stub must not have disturbed them."""

    def test_normalize_offset_weights(self):
        layout = _layout(_NONZERO_TENSION)
        layout_module.NormalizeOffsetWeights(layout)
        for node in layout.nodes.values():
            if node.ConnectedIDs.size:
                assert np.all(node.Weights >= 0.0)
                assert np.all(node.Weights <= 1.0)

    def test_the_configured_floor_is_still_applied(self):
        layout = _layout(_NONZERO_TENSION)
        layout_module.NormalizeOffsetWeights(layout, 0.5, 1.0)
        for node in layout.nodes.values():
            if node.ConnectedIDs.size:
                assert np.all(node.Weights >= 0.5 - 1e-12)

    def test_set_offset_weights_and_population_rank_run(self):
        layout = _layout(_NONZERO_TENSION)
        layout_module.ScaleOffsetWeightsByPopulationRank(layout)
        for node in layout.nodes.values():
            if node.ConnectedIDs.size:
                assert np.all(np.isfinite(node.Weights))

    def test_three_of_the_four_helpers_are_callable(self):
        working = (layout_module.NormalizeOffsetWeights,
                   layout_module.ScaleOffsetWeightsByPopulationRank)
        for helper in working:
            layout = _layout(_NONZERO_TENSION)
            helper(layout)  # must not raise

        with pytest.raises(NotImplementedError):
            layout_module.ScaleOffsetWeightsByPosition(_layout(_NONZERO_TENSION))
