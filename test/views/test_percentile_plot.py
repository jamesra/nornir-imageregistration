import unittest
import os

import numpy as np

import nornir_imageregistration

import setup_imagetest

from nornir_imageregistration.views.alignment_records import *
import nornir_imageregistration


class Test(setup_imagetest.TestBase):

    def test_plotting(self):
        test_data_path = os.path.join(self.TestInputPath, 'Data', 'example_percentile_dataset.npz')
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

        inflection_points = find_inflection_points(percentile, y_fit)
        highest_inflection_point = int(inflection_points[-1])
        cross_products = calculate_deviation(values=percentile_values, above=highest_inflection_point)
        cutoff_percentile_index = np.argmin(cross_products[:, 1]) + highest_inflection_point

        cutoff_value = percentile_values[cutoff_percentile_index]

        second_cutoff = percentile_values[50]

        plot_percentiles(records=values, horz_line_pos_list=[cutoff_value, (
            second_cutoff, {'label': 'Cutoff Value #2', 'color': 'blue', 'linestyle': '--'})])
        self.assertTrue(cutoff_percentile_index == 92)

    def test_repro(self):
        test_data_path = os.path.join(self.TestInputPath, 'Data', 'weight_distance_composite_scores_pass2.npz')
        test_data = np.load(test_data_path)

        # Test data contains
        weight_distance_composite_scores = test_data['weight_distance_composite_scores']

        cutoff_percentile, inflection_percentile, cutoff_value_this_pass, polyfit_weights = nornir_imageregistration.local_distortion_correction.estimate_cutoff(
            weight_distance_composite_scores[:, 0],
            method=nornir_imageregistration.local_distortion_correction.CutoffMethod.Polyfit)

        nornir_imageregistration.views.plot_percentiles(weight_distance_composite_scores[:, 0],
                                                        title=f"Value at percentile",
                                                        horz_line_pos_list=[cutoff_value_this_pass])
