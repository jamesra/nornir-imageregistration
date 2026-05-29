import os

import numpy as np

from nornir_imageregistration.mathfuncs import CutoffMethod, calculate_deviation, estimate_cutoff
import nornir_imageregistration.type_info

import setup_imagetest

from nornir_imageregistration.views.alignment_records import *
import nornir_imageregistration


class Test(setup_imagetest.TestBase):

    def _npz_under_testinput(self, *relative: str) -> str:
        """Path under TESTINPUTPATH; skip if the nornir-testdata fixture is not present."""
        path = os.path.join(self.TestInputPath, *relative)
        if not os.path.isfile(path):
            self.skipTest(
                f"Missing {path!r}. Install or update the nornir-testdata checkout mounted at "
                f"TESTINPUTPATH ({self.TestInputPath!r}); expected files under Data/, "
                "see nornir-imageregistration/tests/views/test_percentile_plot.py."
            )
        return path

    def test_plotting(self):
        test_data_path = self._npz_under_testinput("Data", "example_percentile_dataset.npz")
        test_data = np.load(test_data_path)

        # Test data contains
        percentile = test_data['percentile']
        values = test_data['values']

        percentile_values = np.percentile(values, percentile)
        # Add a polyfit to the linear line
        degree = 5
        coefficients = np.polyfit(percentile, percentile_values, degree)
        # Generate the polynomial function from the coefficients
        polynomial = np.poly1d(coefficients)
        y_fit = polynomial(percentile)

        # Here we find the inflection point with the highest x value.
        # We then calculate the deviation of the values from the line at that point to max(x)
        # The point with the largest deviation has the largest magnitude of cross product.  Negative values are below the line, positive are above.
        inflection_indices, inflection_points = find_inflection_points(percentile, y_fit)
        highest_inflection_point = int(inflection_points[-1])
        cross_products = calculate_deviation(values=percentile_values, above_index=highest_inflection_point)
        cutoff_percentile_index = np.argmax(abs(cross_products[:, 1])) + highest_inflection_point

        cutoff_value = percentile_values[cutoff_percentile_index]

        second_cutoff = percentile_values[50]

        plot_percentiles(records=values, horz_line_pos_list=[cutoff_value, (
            second_cutoff, {'label': 'Cutoff Value #2', 'color': 'blue', 'linestyle': '--'})])
        self.assertTrue(cutoff_percentile_index == 92)

    def test_repro(self):
        test_data_path = self._npz_under_testinput("Data", "weight_distance_composite_scores_pass2.npz")
        test_data = np.load(test_data_path)

        # Test data contains
        weight_distance_composite_scores = test_data['weight_distance_composite_scores']

        cutoff_percentile, inflection_percentile, cutoff_value_this_pass, polyfit_weights = estimate_cutoff(
            weight_distance_composite_scores[:, 0],
            method=CutoffMethod.Polyfit)

        nornir_imageregistration.views.plot_percentiles(weight_distance_composite_scores[:, 0],
                                                        title=f"Value at percentile",
                                                        horz_line_pos_list=[(cutoff_value_this_pass, {})])
