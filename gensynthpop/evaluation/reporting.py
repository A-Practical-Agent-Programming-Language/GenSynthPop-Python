import os
import pandas as pd
from typing import Callable, List, Optional, Tuple

from gensynthpop.evaluation.metrics import (calculate_goodness_of_fit, calculate_z_squared_score,
                                            standardised_absolute_error, total_absolute_error)

ComparisonTuple = Tuple[pd.DataFrame, pd.DataFrame, str, str]


def create_score_table(
        synthetic_frame: pd.DataFrame,
        row_methods: List[Callable[[pd.DataFrame], List[ComparisonTuple]]],
        output_latex_location: Optional[str] = None,
        print_markdown=True,
        print_html=True
) -> None:
    """
    Given a number of methods to score the synthetic population on, concatenates the method results and prints a
    table in markdown and/or html, and optionally writes a LaTeX table to the provided file path.

    The LaTeX table can be included in an existing LaTeX document, encloses in the table environment.

    Each row method takes the synthetic data frame (either synthetic population or synthetic household) and
    return a list of tuples:
        - A dataframe with a column `count` of the number of occurrences of each combination of attributes in the
            synthetic population
        - A dataframe with a column `count` of the number of expected occurrences for the same index
        - The name of the target attribute
        - A string indicating what attributes the target attribute was conditioned on in both data frames.

    Each method may return multiple rows, where each row may contain a different subset of the attributes it is
    conditioned on.

    Args:
        synthetic_frame:            The data frame to evaluate
        row_methods:                List of methods that create score rows based on synthetic data
        output_latex_location:      If provided, location where LaTeX table is written to
        print_markdown:             If true, the score table will be printed to stoud as markdown
        print_html:                 If true, the score table will be printed to stoud as html

    Returns:
        None

    """
    rows = [create_table_row_by_join_on_index(*row) for row_method in row_methods for row in
            row_method(synthetic_frame)]

    df_scores = pd.concat(rows).astype({('', 'DoF'): int})

    if print_markdown:
        print(df_scores.to_markdown())
    if print_html:
        print(df_scores.to_html().replace(
                r'$\times$', '&#215;'
        ).replace(
                '$Z^2$', '<i>Z<sup>2</sup></i>'
        ).replace(
                '$X^2$', '<i>X<sup>2</sup></i>'
        ).replace(
                '$p$', '<i>p</i>'
        ))

    if output_latex_location is not None:
        os.makedirs(os.path.dirname(output_latex_location), exist_ok=True)
        with open(output_latex_location, 'w') as latex_out:
            latex_out.write(df_scores.to_latex(column_format='ll|r|rr|rr|rr', multicolumn_format='|c'))


def export_distributions_from_rows(synthetic_frame: pd.DataFrame,
                                   row_methods: List[Callable[[pd.DataFrame], List[ComparisonTuple]]],
                                   export_directory: str, sep=r' $\latex$ ') -> None:
    """
    Given a synthetic DataFrame and a list of row methods equivalent to those arguments in `create_score_table`,
    exports the CSV files of the joint distributions to the export directory instead of creating a score table.

    See `export_distributions_from_table_row_by_join_on_index` for further details

    Args:
        synthetic_frame:
        row_methods:
        export_directory:
        sep:

    Returns:

    """
    [export_distributions_from_table_row_by_join_on_index(*row, directory=export_directory, sep=sep) for
     row_method in row_methods for row in row_method(synthetic_frame)]


def create_table_row(combined_data: pd.DataFrame, target_attribute: str, jointed_over: str) -> pd.DataFrame:
    """
    Creates a table for one score metric. A table row consists of the following columns:
        - Target attribute
        - (Optional) all conditional attributes
        - The Degrees of Freedom for this combination of attributes
        - The Z-square score and p-value at which this Z-square score would be the critical value in the Chi-square
            distribution
        - The X-square score and p-value at which this X-square score would be the critical value in the Chi-square
            distribution
        - The total and standardized absolute error

    The `combined_data` should contain a data frame with the `count_x` and `count_y` columns, corresponding to the
        contingency table acquired from the synthetic population, and the contingency table from which the target
        attribute was added respectively.

        The frame is indexed by one column for each attribute included in this score

    Args:
        combined_data:      Data frame with `count_x` and `count_y` columns
        target_attribute:   Name of the target attribute
        jointed_over:       List of attributes the target attribute is jointed over for this evaluation row (or empty)

    Returns:

    """
    z_score, z_p, z_dof, *_ = calculate_z_squared_score(combined_data)
    x_score, x_p, x_dof, *_ = calculate_goodness_of_fit(combined_data)

    scores = pd.Series({
        ('', 'DoF'): z_dof,
        ('$Z^2$', 'Score'): z_score,
        ('$Z^2$', '$p$-value'): z_p,
        ('$X^2$', 'Score'): x_score,
        ('$X^2$', '$p$-value'): x_p,
        ('absolute error', 'total'): total_absolute_error(combined_data),
        ('absolute error', 'standardized'): standardised_absolute_error(combined_data)
    })
    scores.name = (target_attribute, jointed_over)
    return scores.to_frame().T


def create_table_row_by_join_on_index(observed: pd.DataFrame, expected: pd.DataFrame, dimension: str,
                                      jointed_over: str) -> pd.DataFrame:
    """
    Given two data frames with the same index and a `count` column in both, creates a score row (see `create_table_row`)

    Also creates a CSV file containing the values uses to determine the scores for this table row.

    Args:
        observed:
        expected:
        dimension:
        jointed_over:

    Returns:

    """
    df_joint = observed.merge(expected, left_index=True, right_index=True, how='outer')

    return create_table_row(df_joint, dimension, jointed_over)


def export_distributions_from_table_row_by_join_on_index(
        observed: pd.DataFrame, expected: pd.DataFrame, dimension: str, jointed_over: str, directory: str,
        sep=r' $\times$ ') -> None:
    """
    Takes the same arguments as `create_table_row_by_join_on_index`, but stores the result into a CSW file in the
    output directory instead of attempting to create a score table row.

    Assumes the conditional attributes can be split by the `sep` value (default: $\times$ because of LaTeX)

    Args:
        observed:
        expected:
        dimension:
        jointed_over:
        directory:
        sep:

    Returns:

    """
    os.makedirs(directory, exist_ok=True)
    df_joint = observed.merge(expected, left_index=True, right_index=True, how='outer')
    csv_name = dimension.replace(" ", "_")
    if jointed_over:
        csv_name += "_conditioned_on_"
        csv_name += jointed_over.replace(sep, "_and_").replace(" ", "_")
    df_joint.rename(columns={
        'count_x': 'synthetic_population_counts',
        'count_y': 'fitted_source_data_counts'
    }).to_csv(os.path.join(directory, f'{csv_name}.csv'))
