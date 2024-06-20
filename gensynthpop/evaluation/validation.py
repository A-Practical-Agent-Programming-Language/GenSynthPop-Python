import warnings
from typing import List, Union

import pandas as pd

from gensynthpop.evaluation.metrics import calculate_z_squared_score
from gensynthpop.utils.extractors import synthetic_population_to_contingency


def validate_fitted_distribution(df: pd.DataFrame, margins: Union[pd.Series, pd.DataFrame],
                                 margin_names: Union[str, List[str]],
                                 joint_distribution_name: str) -> None:
    """
    Used to validate the joint distribution of two or more categories after applying iterative proportional fitting
    to the margins (i.e., the categories) the data was fitted to.

    Uses a statistical test by comparing the Z-squared score to a chi-squared distribution.

    A p-value > 0.05 is traditionally considered acceptable in model testing.

    Args:
        df:                         The fitted distribution
        margins:                    The table representing the margins to validate
        margin_names:               The names of those margins to validate
        joint_distribution_name:

    Returns:

    """

    if isinstance(margins, pd.Series):
        margins.index.name = tuple(margin_names)
        margins.rename("count")
    else:
        margins = margins[margin_names + ["count"]].set_index(margin_names)

    if isinstance(margin_names, str):
        margin_names = [margin_names]

    joint_attributes = " X ".join(margin_names)

    df_combined = df.groupby(margin_names).sum()[["count"]].merge(margins, left_index=True, right_index=True)
    z, p, _, _, _ = calculate_z_squared_score(df_combined)

    if p < 0.05:
        warnings.warn(
                f"The fitted joint distribution of {joint_distribution_name} does not statistically fit the margins"
                f"of {joint_attributes}. p-value was p = {p} (Z² = {z}). Caution advised when using this fitted "
                f"distribution")
    else:
        print(f"Fitted {joint_distribution_name} fits margins of {joint_attributes} (p = {p}, Z² = {z})")


def validate_synthetic_population_fit(synthetic_population: pd.DataFrame, expected: pd.DataFrame, dimensions: List[str],
                                      name: str) -> None:
    """
    We use a statistical test (Z-squared score evaluated against a chi-squared distribution) to verify if the
    distribution of the listed dimensions in the synthetic population matches the expected distribution of those
    attributes.

    A p-value indicating the chance a Z-squared score as or more extreme than the obtained score would occur if the
    two distributions are in fact the same is given. Traditionally, a value of p < 0.05 means that probability is so
    small the null-hypothesis that the two distributions are the same would be rejected in favour of the alternative
    hypothesis that they are actually different. In model testing, conversely, a p value >= 0.05 indicates a good fit.

    Args:
        synthetic_population:   The synthetic population containing the dimensions.
        expected:               The expected (joint) distribution containing the dimensions.
        dimensions:     List of all categorical variables (appearing in both tables) to include in the statistical test.
        name:           Name of the attribute that was added.

    Returns:

    """
    df_combined = synthetic_population_to_contingency(
            synthetic_population,
            dimensions
    ).reset_index().merge(
            expected[dimensions + ["count"]],
            on=dimensions
    )
    _, p, _, _, _ = calculate_z_squared_score(df_combined)
    joint_variables = " X ".join([d for d in dimensions if d != name])
    if p < 0.05:
        warnings.warn(
                f"After adding {name}, distribution of {name} over {joint_variables} in the synthetic population does "
                f"not match known margins. p-value was {p}. Caution advised when using this synthetic population")
    else:
        print(f"{name} over {joint_variables} is from same distribution (p = {p})")
