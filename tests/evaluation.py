# This file verifies that the metrics as implemented are consistent with the scores those metrics should generate
# as reported by Voas & Williamson (2000) in their article:
# Voas, D., & Williamson, P. (2000). An evaluation of the combinatorial optimisation approach to the creation of
# synthetic microdata. International Journal of Population Geography, 6(5),

import unittest
from typing import List

import pandas as pd

from gensynthpop.evaluation.metrics import (_calculate_one_z_squared_score, calculate_goodness_of_fit,
                                            calculate_z_squared_score, total_absolute_error)

# Table 1 from Voas & Williamson (2000)
df_table_1 = pd.DataFrame(data=dict(
        expected=[2, 2, 2, 2, 1, 1, 1],
        observed_minus_expected=[-2, -1, 1, 2, -1, 1, 2],
        from_binomial=[-1.11, -0.24, 0.46, 1.07, -0.34, 0.63, 1.41],
        using_continuity_correction=[-1.07, -0.36, 0.36, 1.07, -0.50, 0.50, 1.51],
        unmodified=[-1.42, -0.71, 0.71, 1.42, -1.00, 1.00, 2.01]
))

# Table 2 from Voas & Williamson (2000)
df_table_2 = pd.DataFrame([
    (50, 55, "0.81", "0.45", "0.5"),
    (5, 10, "4.26", "4.09", "5.0"),
    (0, 2, "4.04", "4.01", "4.0"),
    (3, 6, "2.15", "2.1", "3.0"),
    (40, 50, "3.76", "2.45", "2.5"),
    (4, 5, "0.07", "0.06", "0.25"),
    (12, 3, "6.84", "6.17", "6.75"),
    (12, 21, "6.84", "6.17", "6.75"),
    (24, 5, "18.8", "15.0", "15.0"),
    (10, 5, "2.25", "2.07", "2.5"),
    (10, 8, "0.25", "0.23", "0.4")
], columns=["count_y", "count_x", "Z_n100", "Z_n500", "X_squared"])

table_2_totals = dict(Z_n100=50.1, Z_n500=42.8, X_squared=46.7)


# Voas & Williamson (2000):
# Voas, D., & Williamson, P. (2000). An evaluation of the combinatorial optimisation approach to the creation of
# synthetic microdata. International Journal of Population Geography, 6(5), 349-366.

class TestEvaluation(unittest.TestCase):

    def test_absolute_error(self):
        with self.subTest("Identical counts means no error"):
            df = make_df([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
            self.assertEqual(0, total_absolute_error(df))

        with self.subTest("Errors in both directions work adds up the absolutes"):
            df = make_df([0, 10, 10, 10, 10, 10, 10], [10, 9, 8, 11, 12, 0, 10])
            self.assertEqual(26, total_absolute_error(df))

    def test_calculate_goodness_of_fit(self):
        with self.subTest("Identical value counts means perfect fit"):
            df = make_df([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
            x_squared, p_value, _dof, _ = calculate_goodness_of_fit(df)
            self.assertEqual(0, x_squared)
            self.assertEqual(1, p_value)
            self.assertEqual(5, _dof)

        with self.subTest("Near identical values means good fit with non-zero score"):
            df = make_df([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9, 9.1, 9.9])
            x_squared, p_value, dof, _ = calculate_goodness_of_fit(df)
            self.assertAlmostEqual(0.028583, x_squared)
            self.assertAlmostEqual(1, p_value)
            self.assertEqual(10, dof)

        with self.subTest("Arbitrarily different DF should have high score and low p-value"):
            df = make_df([0, 10, 10, 10, 10, 10, 10], [10, 9, 8, 11, 12, 0, 10])
            x_squared, p_value, dof, _ = calculate_goodness_of_fit(df)
            self.assertAlmostEqual(111.03535353535354, x_squared)
            self.assertAlmostEqual(0, p_value)
            self.assertEqual(7, dof)

        for _, row in df_table_2.iterrows():
            with self.subTest(
                    f"X^2 according to Voas & Williamson (2000), actual/synthetic = {row.count_y}/{row.count_x}"):
                df = row.to_frame().T[["count_x", "count_y"]]
                x_squared, *_ = calculate_goodness_of_fit(df)
                expected_str = row[f'X_squared']
                precision = len(expected_str.split(".")[1])
                self.assertEqual(float(expected_str), round(x_squared, precision))

        with self.subTest(f"Total X^2 should sum to {table_2_totals[f'X_squared']}"):
            df = df_table_2[["count_x", "count_y"]]
            x_squared, *_ = calculate_goodness_of_fit(df)
            self.assertEqual(table_2_totals[f'X_squared'], round(x_squared, 1))

    def test_calculate_z_squared_score(self):
        with self.subTest("Identical value counts means perfect fit, but with continuity factor error"):
            df = make_df([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
            x_squared, p_value, _dof, *_ = calculate_z_squared_score(df)
            self.assertAlmostEqual(0.6572420634920635, x_squared)
            self.assertAlmostEqual(0.9852326670691531, p_value)
            self.assertEqual(5, _dof)

        with self.subTest("Near identical values means good fit with non-zero score"):
            df = make_df([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9, 9.1, 9.9])
            x_squared, p_value, dof, *_ = calculate_z_squared_score(df)
            self.assertAlmostEqual(0.4897587276878192, x_squared)
            self.assertAlmostEqual(0.9999940128959484, p_value)
            self.assertEqual(10, dof)

        with self.subTest("Arbitrarily different DF should have high score and low p-value (<0.05)"):
            df = make_df([0, 10, 10, 10, 10, 10, 10], [10, 9, 8, 11, 12, 0, 10])
            x_squared, p_value, dof, *_ = calculate_z_squared_score(df)
            self.assertAlmostEqual(113.17431853711219, x_squared)
            self.assertEqual(0.0, p_value)
            self.assertEqual(7, dof)

        # Remainder validates Z^2 (N = 100) and Z^2 (N = 500) values from Voas & Williamson paper (Table 2)
        for n in [100, 500]:
            for _, row in df_table_2.iterrows():
                with self.subTest(
                        f"William & Voas paper, table 2, n = {n}, actual/synthetic = {row.count_y}/{row.count_x}"):
                    df = row.to_frame().T[["count_x", "count_y"]]
                    df.loc[:, ["observed_total", "expected_total"]] = n, n
                    z, *_ = calculate_z_squared_score(df)
                    expected_str = row[f'Z_n{n}']
                    precision = len(expected_str.split(".")[1])
                    self.assertEqual(float(expected_str), round(z, precision))

            with self.subTest(f"Total Z^2 (n={n}) should sum to {table_2_totals[f'Z_n{n}']}"):
                df = df_table_2[["count_x", "count_y"]]
                df.loc[:, ["observed_total", "expected_total"]] = n, n
                z, *_ = calculate_z_squared_score(df)

                # Note, paper says Z-squared values for N = 100 sum to 50.1, but it is unclear if this is a rounding
                # error. All other values in the column (the ones that are summed up) are correct within the given
                # precision
                if n != 100:
                    self.assertEqual(table_2_totals[f'Z_n{n}'], round(z, 1))

    def test_calculate_one_z_squared_score(self):
        """
        The results have been confirmed with Table 1 from the Voas & Williamson paper, by overriding the
        observed_total and expected_total values with 150.

        Returns:

        """

        df_table_1.loc[:, "observed"] = df_table_1.observed_minus_expected + df_table_1.expected

        for _, row in df_table_1.iterrows():
            with self.subTest(
                    f"Z-score for small cells where N = 150 (observed/expected = {row.observed}/{row.expected}"):
                df = pd.Series(
                        data=dict(count_x=row.observed, count_y=row.expected, observed_total=150, expected_total=150))
                new_row = _calculate_one_z_squared_score(df)
                self.assertEqual(row.using_continuity_correction, round(new_row.z, 2))


def make_df(count_x: List[float], count_y: List[float]) -> pd.DataFrame:
    return pd.DataFrame(data=dict(count_x=count_x, count_y=count_y))
