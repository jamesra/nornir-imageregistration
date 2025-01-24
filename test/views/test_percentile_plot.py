import unittest
import os

import numpy as np

import nornir_imageregistration

import setup_imagetest

from nornir_imageregistration.views.alignment_records import *


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
        cross_products = find_maximum_deviation(records=percentile_values)
        cutoff_percentile_index = np.argmin(cross_products[highest_inflection_point:, 1]) + highest_inflection_point
        self.assertTrue(cutoff_percentile_index == 82)

        cutoff_value = percentile_values[cutoff_percentile_index]

        plot_percentiles(records=values, horz_line_pos_list=[cutoff_value])
