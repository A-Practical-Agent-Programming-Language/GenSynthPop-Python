from typing import List

import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, chisquare, norm


def total_absolute_error(_df: pd.DataFrame) -> float:
    """
    Calculates the absolute difference between the observed and expected counts.
    Scales the expected counts so its totals match the totals of the observed counts if the difference between
    expected and observed totals is more than 5%

    Args:
        _df: Data frame to test

    Returns:
        Total absolute error of this distribution
    """
    _df = _df.copy()
    if abs(_df.count_x.sum() - _df.count_y.sum()) / np.mean([_df.count_x.sum(), _df.count_y.sum()]) > 0.05:
        print("Scaling expected values before calculating total absolute error")
        _df.count_y = _df.count_y.transform(lambda x: x / _df.count_y.sum() * _df.count_x.sum())
    _df["abs_diff"] = abs(_df.count_x - _df.count_y)
    return _df.abs_diff.sum()


def standardised_absolute_error(_df: pd.DataFrame) -> float:
    """
    The SAE is the TAE divided by the total expected count.
    This allows for easier comparison between different synthetic population files

    Args:
        _df: Data frame to test

    Returns:
        Standardized absolute error of the distribution

    """
    return total_absolute_error(_df) / _df.count_y.sum()


def percentage_points_difference(_df: pd.DataFrame) -> float:
    """
    Calculates the sum of the absolute difference in fractions in each group. No need to scale.

    Note: Corresponds to the Standardized Absolute Error when observed and expected totals are the same

    Args:
        _df: Data frame to test

    Returns:
        Absolute difference between the distribution and the target

    """
    _df = _df.reset_index()
    if "neighb_code" in _df.columns:
        _df['total_x'] = _df.groupby('neighb_code').count_x.transform('sum')
        _df['total_y'] = _df.groupby('neighb_code').count_y.transform('sum')
        _df['count_y'] = _df.count_y / _df.total_y * _df.total_x
    else:
        _df['total_x'] = _df.count_x.sum()
        _df['total_y'] = _df.count_y.sum()
    _df["frac_x"] = _df.count_x / _df.total_x
    _df["frac_y"] = _df.count_y / _df.total_y
    _df["abs_diff"] = abs(_df.frac_x - _df.frac_y)
    return _df.abs_diff.sum()


def calculate_goodness_of_fit(_df: pd.DataFrame) -> (float, float, float, List[float]):
    """
    Calculates the Pearson's Chi-square goodness-of-fit

    In D. Voas & P. Williamson:
        0 is replaced by 1 in the denominator (but not the numerator) of $X^2$ and Z.
        (From Table 2 caption, pp. 188)

    Args:
        _df: Data frame to test

    Returns:
        Tuple of X-square score, p-value, number of degrees of freedom, and the critical value at the alpha levels

    """
    _dof = _df.shape[0]  # - 1  ##  Note, according to Voas and Williamson, the last cell is still a free choice
    critical_values = chi2.ppf(1 - CRITICAL_ALPHA, df=_dof)
    enumerator = np.square(_df.count_x - _df.count_y)
    denominator = _df.count_y.replace(0, 1)
    ratios = enumerator / denominator
    p_value = 1 - chi2.cdf(ratios.sum(), df=_dof)
    return ratios.sum(), p_value, _dof, critical_values


@DeprecationWarning
def calculate_independence(_df: pd.DataFrame) -> (float, float, List[float]):
    """
    Performs the Pearson's Chi-square test of independence.
    Note: This is NOT THE RIGHT METRIC to use, as it tests the independence between two attributes and does not
    compare distributions

    Args:
        _df: Data frame to test

    Returns:
        Tuple of X-square score, number of degrees of freedom, and the critical value at the alpha=0.05 level

    """
    x_square, p, _dof, _ = chi2_contingency(_df)
    return x_square, _dof, chi2.ppf(1 - CRITICAL_ALPHA, df=_dof)


def calculate_z_squared_score(df: pd.DataFrame) -> (float, float, float, List[float], pd.DataFrame):
    """
    Calculates the Z-score (which can be used as the X-squared value in a Chi^2 test)

    In D. Voas & P. Williamson:
        - 0 is replaced by 1 in the denominator (but not the numerator) of $X^2$ and Z.
        (From Table 2 caption, pp. 188)
        - Continuity correction is applied except where the expected amount is zero (pp. 188)
            - Equation on pp. 183:
                Using a continuity correction factor in the numerator of the test statistic is of considerable benefit
                (in relation to using a continues normal distribution as an approximation to the discrete binomial).
                The quantity 1/2N is simply subtracted from a non-zero difference in proportions tij-pij or added if
                the difference is negative (Fleiss, 1981, p. 13)
            - "Continuity correction is applied except where the expected amount is zero." (p. 187)

        See tests

    Args:
        df: Data frame to test

    Returns:
        Tuple of X-square score, number of degrees of freedom, and the critical value at the alpha=0.05 level

    """
    _df = df.copy()

    if "observed_total" not in _df.columns:
        _df.loc[:, "observed_total"] = _df.count_x.sum()
    if "expected_total" not in _df.columns:
        _df.loc[:, "expected_total"] = _df.count_y.sum()

    _df = _df.apply(_calculate_one_z_squared_score, axis=1)
    _df.loc[:, "p"] = 1 - norm().cdf(_df.z)
    _dof = _df.shape[0]
    z_square = _df.z.apply(lambda x: x * x).sum()
    p_value = 1 - chi2.cdf(z_square, df=_dof)
    critical_values = chi2.ppf(1 - CRITICAL_ALPHA, df=_dof)

    return z_square, p_value, _dof, critical_values, _df


def _calculate_one_z_squared_score(row):
    r"""
    This implements the Z-score with a continuity factor as proposed by Fleiss, 1981, p. 13.
    Equation taken from Voas & Williamson p. 183

    $Z = \frac{(t_{ij} - p_{ij}) +/- \frac{1}{2N}}{\sqrt{p_{ij}(1-p_{ij})}{N}}$

    Where:
        T_ij is the observed count (the value in the ijth cell)
        P_ij is the expected count
        N is expected total (sometimes, N_t and N_e are distinguished)
        t_{ij} = T_{ij}/N (observed proportion)
        p_{ij} = P_{ij}/N (expected proportion)


    Thus the observed value itself can be used as an approximat e Z score in the case
    where the expected value is 0. In practice we can avoid the simplifying assumptionsGoodness-of-Fi t Measures
    185
    by calculating Z in the normal way, except with pij 5 1/N (rather than 0/N) in the
    denominator. It is probably not desirable to apply the continuity correction in this
    case, since replacing zero with a surrogate value serves in eVect to reduce Z, and a
    further reduction would be unwarranted.

    Args:
        row:

    Returns:

    """
    observed_proportion = row.count_x / row.observed_total
    expected_proportion = row.count_y / row.expected_total
    enumerator = (observed_proportion - expected_proportion)
    if row.count_y != 0:
        continuity_correction_factor = (1 / (2 * row.expected_total))  # Fleiss, 1981, p. 13
        if row.expected_total != 0:
            if enumerator >= 0:
                enumerator -= continuity_correction_factor
            elif enumerator < 0:
                enumerator += continuity_correction_factor

    if row.count_y == 0:  # Expected value is 0
        expected_proportion = 1 / row.expected_total

    denominator = np.sqrt(expected_proportion * (1 - expected_proportion) / row.expected_total)
    row["z"] = enumerator / denominator
    return row


def score_distribution(_df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a data frame with two columns, count_x (observed counts) and count_y (expected counts), with the groups
    they count as their index.
    Calculates the Z-score, goodness of fit and test of independence against the appropriate $\chi^2$ critical values.
    Returns a DF with the most important statistics for this dataframe for each of those three tests

    Args:
        _df:

    Returns:
        Dataframe summarizing the statistical test metrics of the distribution

    """
    _df = _df.copy()
    headers_l1 = ["X²", "DoF", "p"]
    headers_l2 = ["", "", ""]
    for alpha in CRITICAL_ALPHA:
        headers_l1 += [f"α={alpha}"] * 2
        headers_l2 += ["Critical Value", "Accepted"]

    index = pd.MultiIndex.from_tuples(list(zip(headers_l1, headers_l2)))

    index_names = list()
    data = list()
    for name, method in STATISTICAL_TESTS.items():
        score, p, dof, critical_values, *_ = method(_df)
        _data = [score, dof, p]
        for alpha, critical_value in zip(CRITICAL_ALPHA, critical_values):
            _data += [critical_value, score < critical_value]
        index_names.append(name)
        data.append(_data)

    df_results = pd.DataFrame(
            data,
            index=index_names,
            columns=index
    )

    df_results_extra = pd.DataFrame(data={
        ("X²", ""): [
            # percentage_points_difference(_df),
            total_absolute_error(_df),
            standardised_absolute_error(_df)
        ]
    }, index=["Total Absolute Error", "Standardized Absolute Error"])
    df_results = pd.concat([df_results, df_results_extra])
    return df_results


CRITICAL_ALPHA = np.array([0.05])  # , .95, .999, .99999])

STATISTICAL_TESTS = {
    "Z-score": calculate_z_squared_score,
    "Goodness of Fit": calculate_goodness_of_fit,
    # "Test of Independence": calculate_independence
}

if __name__ == "__main__":
    df = pd.DataFrame(data=dict(
            count_x=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            count_y=[1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9, 9.1, 9.9]
    ), index=["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"])
    print(score_distribution(df))
    r1 = chisquare(f_obs=df.count_x, f_exp=df.count_y)
    r2 = calculate_goodness_of_fit(df)
    r3 = calculate_z_squared_score(df)
    print(r1.statistic, r2[0], r1.statistic == r2[0])
    print(r1.pvalue, r2[1], r1.pvalue == r2[1])
    print(r3[0], r3[1])
