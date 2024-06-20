from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, cast

import pandas as pd

from gensynthpop.evaluation.metrics import calculate_z_squared_score
from gensynthpop.utils.extractors import synthetic_population_to_contingency
from ipfn.ipfn import ipfn


class ConditionalAttributeAdder:

    def __init__(
            self,
            df_synthetic_population: pd.DataFrame,
            df_contingency: pd.DataFrame,
            target_attribute: str,
            group_by: Optional[List[str]] = None,
    ):
        """

        Args:
            df_synthetic_population:
            df_contingency:     Distribution from which the target attribute is to be added
            target_attribute:   Name of attribute that is to be added. Should be present in df_contingency
            group_by:           Attributes appearing both in the df_contingency and already in the synthetic population
        """
        self.df = df_synthetic_population.reset_index()
        if "index" in self.df.columns:
            self.df.drop("index", axis=1, inplace=True)
        if "level_0" in self.df.columns:
            self.df.drop("level_0", axis=1, inplace=True)
        self.df_contingency = df_contingency
        self.target_attribute = target_attribute
        self.group_by = group_by
        self.margins: Optional[List[pd.DataFrame]] = None
        self.margins_names: Optional[List[List[str]]] = None
        self.margins_group: Optional[List[str]] = self.group_by

    def add_margins(
            self,
            margins: List[pd.DataFrame],
            margins_names: List[List[str]]
    ) -> ConditionalAttributeAdder:
        """
        If no data is available for each group as resulting from the group_by class argument, the data may have to
        be fitted to each group separately. For this IPF (Iterative Proportional Fitting) can be used, provided
        that the margins of each group are available. This step can be enabled by providing those margins in
        with this method

        Args:
            margins:        List of dataframes containing margins that the df_contingency should be fitted to
                            for each group
            margins_names:  Names of the margins (as they appear in the df_contingency) provided in
                            `margins`

        Returns:

        """
        if margins is None or margins_names is None:
            raise ValueError("When either margins_names or margins_data_frames argument is provided, the other is "
                             "required as well")
        if len(margins) != len(margins_names):
            raise ValueError("margins_data_frames and margins_names have to be the same length")

        self.margins = margins
        self.margins_names = margins_names
        self.margins_group = list(
                set([m for m_list in self.margins_names for m in m_list if m in self.df.columns])
        )

        return self

    def run(self) -> pd.DataFrame:
        """
        Runs the attribute adder to add a new target attribute to the synthetic population conditioned on the
        conditional attributes in the source distribution

        Each attribute in `group_by` needs to be present in both the synthetic population and in all the
        `margins_data_frames`

        Returns:

        """

        # Add new empty column
        self.df[self.target_attribute] = None
        self.df.astype({self.target_attribute: 'object'})

        # Iterate over all groups
        for group_name, group in self.df.groupby(self.group_by):
            group_name = cast(Tuple[str], group_name)
            df_contingency_group = self.__ipf_fit_contingency_table(group_name)
            group_fractions = self.__get_group_fractions(df_contingency_group)
            if not self.margins:
                group_values = _get_agent_values_from_fractions(group_fractions, group.shape[0])
                mask = _get_group_mask(self.df, group_name, self.group_by)
                self.df.loc[mask, [self.target_attribute]] = group_values
            else:
                sub_group_by = [g for g in self.margins_group if g not in self.group_by and g != self.target_attribute]
                for sub_group_name, sub_group in self.df.groupby(sub_group_by):
                    sub_group_name = cast(Tuple[str], sub_group_name)
                    mask = _get_group_mask(self.df, group_name, self.group_by)
                    mask &= _get_group_mask(self.df, sub_group_name, sub_group_by)

                    if not mask.any():
                        # No need to add attributes to non-existent agents, plus will run into some zero division errors
                        # if we continue
                        continue

                    group_fractions_as_df = pd.DataFrame(group_fractions).reset_index()
                    other_mask = _get_group_mask(group_fractions_as_df, sub_group_name, sub_group_by)
                    sub_group_fractions = group_fractions_as_df[other_mask].set_index(self.target_attribute)[
                        "fraction"]

                    if not sub_group_fractions.index.is_unique:
                        # Group by index in case index contains duplicates (which can be the case when margins are less
                        # precise than target attribute)
                        print(f"Attribute {sub_group_fractions.index.name} is not unique. Grouping.")
                        sub_group_fractions = sub_group_fractions.groupby(level=0).sum()

                    group_values = _get_agent_values_from_fractions(sub_group_fractions, self.df[mask].shape[0])
                    self.df.loc[mask, [self.target_attribute]] = group_values

        self.verify_target_attribute()

        return self.df

    def __ipf_fit_contingency_table(self, group_name: Tuple[str]) -> any:
        """
        Returns the relevant part of the target contingency table for this group name (containing the value of the
        group-by clause).

        If separate margins are provided, this group is additionally fitted to those margins

        Args:
            group_name:

        Returns:

        """
        if not self.margins:
            mask = _get_group_mask(self.df_contingency, group_name, self.group_by)
            return self.df_contingency[mask]

        aggregates = list()
        for df, dimension in zip(self.margins, self.margins_names):
            mask = _get_group_mask(df, group_name, self.group_by)
            df_masked = df[mask]
            count = df_masked.set_index(dimension)["count"].astype('float')  # converts to Series
            aggregates.append(count)

        df_to_fit = self.df_contingency.copy()
        ipf = ipfn(df_to_fit, aggregates, self.margins_names, weight_col="count")
        df_fitted = ipf.iteration()

        return df_fitted

    def __get_group_fractions(self, df: pd.DataFrame) -> pd.Series:
        """
        Calculates fractions from the contingency data frame

        Returns:
        """
        df_fractions = self._calculate_fractions(df)
        index = [self.target_attribute]
        if self.margins_names:
            index += [margin_name for margin_name_group in self.margins_names for margin_name in margin_name_group]
        index = list(set(index))  # Deduplication of index names
        df_fractions = df_fractions.set_index(index)['fraction']
        return df_fractions

    def _calculate_fractions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes a contingency table with n conditional attributes and a `count` column as input, and
        adds a column `fraction` containing what fraction the count value is of the total count within each group of
        conditional attributes.

        Used to determine what fractions of agents in the synthetic population with the same values for all the
        conditional attributes should have what value of the target attribute.

        Returns:

        """
        df_fractions = df.copy()
        group_by = self.group_by.copy()
        if self.margins_names:
            group_by += self.margins_group
        group_by = [gb for gb in group_by if gb in df.columns and gb != self.target_attribute]
        fractions = df.groupby(group_by)["count"].transform(lambda x: x / x.sum() if x.sum() != 0 else 0)

        df_fractions.loc[:, ["fraction"]] = fractions
        return df_fractions

    def verify_target_attribute(self):
        # Do some checks
        if not self.df[self.target_attribute].notnull().all():
            warnings.warn(
                    f"There was an issue conditionally assigning the {self.target_attribute} value that caused not all"
                    f" agents having been assigned a value. Please refrain from using the result")

        group_by = self.margins_group + [self.target_attribute]

        # Check significance of target attribute
        _, p, _, _, _ = calculate_z_squared_score(
                synthetic_population_to_contingency(
                        self.df, group_by
                ).merge(self.df_contingency, on=group_by).set_index(group_by)
        )
        if p < 0.05:
            warnings.warn(
                    f"The attribute {self.target_attribute} was added, but its distribution in the synthetic population"
                    f" does not statistically match the contingency table this attribute was added from. "
                    f" P-value was {p}."
                    " Caution advised when using this fitted distribution")


def _calculate_group_counts(df_fractions: pd.Series, n_agents_total) -> pd.Series:
    """
    Takes a data frame as generated by @code{ConditionalAttributeAdder._calculate_fractions} and returns for each 
    combination of conditional attribute values the total number of synthetic agents that should have each of the 
    target attribute values, as based on the total number of agents `n_agents_total` in the synthetic population.

    This would be a simple transform were it not for the fact that we have to map the fractions to integer counts of
    agents. This method tries to find the group with the largest fractional difference in agent counts, and adjusts
    that group accordingly, until the sum of the counts equals the n_agents_total variable.

    Args:
        df_fractions:
        n_agents_total: Total number of synthetic agents

    Returns:

    """
    df_fractions = df_fractions.copy()

    # Apply the fraction to the total number of agents to obtain a first estimate of synthetic agent counts per group
    counts = (df_fractions * n_agents_total).transform(round)

    while (group_total := counts.sum()) != n_agents_total:
        # Find the group with the largest fractional difference to the expected fraction
        differences = ((counts / n_agents_total) - df_fractions).transpose()
        correction_target = differences.idxmin() if group_total < n_agents_total else differences.idxmax()

        # Adjust that group in the direction that moves the count towards `n_agents_total`
        if group_total < n_agents_total:
            counts[correction_target] += 1
        else:
            counts[correction_target] -= 1
    return counts


def _get_agent_values_from_fractions(group_fractions: pd.Series, group_agent_count: int) -> List:
    """
    Calculates the relevant counts of agents based on the fractions

    Args:
        group_fractions:
        group_agent_count:

    Returns:

    """
    group_counts = _calculate_group_counts(group_fractions, group_agent_count)
    group_values = list()
    for attr_value, series in group_counts[group_counts.notna()].items():
        group_values += [attr_value] * series
    return group_values


def _get_group_mask(df: pd.DataFrame, group_name: Tuple[str],
                    group_by: Optional[List[str]]) -> pd.Series:
    """

    Args:
        df:
        group_name:
        group_by:

    Returns:

    """
    mask = pd.Series([True] * df.shape[0])
    for attr, value in zip(group_by, group_name):
        if attr in df.columns:
            mask &= df[attr] == value
    return mask
