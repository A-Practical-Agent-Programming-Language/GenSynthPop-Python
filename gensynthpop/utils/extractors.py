import sys
import warnings
from typing import List, Optional

import pandas as pd


def synthetic_population_to_contingency(df_synthetic_population: pd.DataFrame,
                                        columns: Optional[List[str]], full_crostab: bool = False) -> pd.DataFrame:
    """
    A synthetic population is a data frame that contains one row for each agent.
    This method creates a contingency table from the synthetic population, listing for each combination of
    attributes how many agents have those attributes.

    Args:
        df_synthetic_population:    Synthetic Population Dataframe
        columns:                    List of column names (i.e. attributes) to consider
        full_crostab:               Fills missing combinations with a 0 value if set to true. Omits missing combinations
                                    otherwise (default)

    Returns:

    """
    if columns is None:
        columns = df_synthetic_population.columns
    index = [df_synthetic_population[col_name] for col_name in columns]
    df = pd.crosstab(index, columns=["count"])
    df.columns.name = None
    if full_crostab:
        if not isinstance(df.index, pd.MultiIndex):
            unstacked = df.unstack(fill_value=0.)
            df = unstacked.stack()
        else:
            levels = [df.index.get_level_values(i).unique() for i in range(df.index.nlevels)]
            new_index = pd.MultiIndex.from_product(levels)
            df = df.reindex(new_index, fill_value=0.)

    return df


def multicolumn_to_attribute_values(df: pd.DataFrame, attr_name: str, columns: List[str]) -> pd.DataFrame:
    """
    Wrapper method for @code{pd.melt} to convert a data frame where each attribute value has its own column which
    contains
    the value counts for that attribute, to a data frame where one column contains the attribute values and a count
    column is added listing the number of agents having that attribute

    Args:
        df:         Dataframe to convert
        attr_name:  Name of the attribute that is combined
        columns:    All columns representing values for the attribute

    Returns:

    """
    id_vars = [col_name for col_name in df.columns if col_name not in columns]
    return pd.melt(df, id_vars=id_vars, value_vars=columns, var_name=attr_name, value_name='count')


def age_to_age_group(age: int, age_groups: List[str]):
    """
    Takes an age and an array of age groups, and maps the age to one of the age groups.

    An age group is denoted as a string denoted as one of:
        "{i}-{j}", where i is an integer lower bound (inclusive) of the age group, and {j} is an integer upper
            bound (exclusive) of the age group;
        or "{i}+" where {i} is an inclusive upper bound

    At most one age group may contain a plus ("+") character.
    Age groups must fully cover the expected range, but may not be overlapping. As such, a valid example of an age group
    is:     ["0-15", "15-25", "25-45", "45-65", "65+"]


    Args:
        age:        Integer age
        age_groups: List of age groups following the syntax specified above

    Returns:
        The age group this integer age belongs in
    """
    for age_group in sorted(age_groups):
        if "+" in age_group:
            lower_bound = int(age_group[:-1])
            upper_bound = sys.maxsize
        else:
            lower_bound, upper_bound = [int(x) for x in age_group.split("-")]
        if lower_bound <= age < upper_bound:
            return age_group

    warnings.warn(f"No age group found for integer age \"{age}\" in age groups {age_groups}")


def get_margin_series_from_synthetic_population(df_synth_pop: pd.DataFrame, margins: List[List[str]]):
    return {
        tuple(names): synthetic_population_to_contingency(df_synth_pop, names, len(names) > 1)["count"]
        for names in margins
    }


def get_margin_frames_from_synthetic_population(df_synth_pop: pd.DataFrame, margins: List[List[str]]):
    return {
        tuple(names): synthetic_population_to_contingency(df_synth_pop, names, len(names) > 1).reset_index()
        for names in margins
    }
