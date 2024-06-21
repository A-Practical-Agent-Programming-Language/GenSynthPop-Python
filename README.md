[![DOI](https://zenodo.org/badge/810318631.svg)](https://zenodo.org/doi/10.5281/zenodo.11474109)

# GenSynthPop

This repository contains the implementation of GenSynthPop,
a sample-free tool to construct Synthetic Populations and Households from mixed-aggregation contingency tables

The work in this repository is described in
*GenSynthPop: Generating a Spatially Explicit Synthetic Population of Agents and Households from Aggregated Data*[^1].

[^1]: Marco Pellegrino, Jan de Mooij, Tabea Sonnenschein et al. *GenSynthPop: Generating a Spatially Explicit Synthetic
Population of Agents and Households from Aggregated Data*, 09 October 2023, PREPRINT (Version 1) available at Research
Square [https://doi.org/10.21203/rs.3.rs-3405645/v1]

An R implementation of this library is available [here](https://github.com/TabeaSonnenschein/GenSynthPop)

A reference implementation is available
[here](https://github.com/A-Practical-Agent-Programming-Language/Synthetic-Population-The-Hague-South-West)

## Intuition

This library allows generating a synthetic population from contingency tables and marginal data one attribute at the
time. It does not assume the presence of a detailed sample. Refer to the
[reference implementation](https://github.com/A-Practical-Agent-Programming-Language/Synthetic-Population-The-Hague-South-West)
for details.

Generally when data is published there is a trade-off in accuracy between the joint distribution of attributes and the
spatial granularity. Detailed data may be available for very small regions, but only contain a few or even one
attribute.
In order to achieve the best spatial heterogeneity, one would prefer to use those data. However, in order to get an
accurate representation of the population as a whole, the joint distirbution of attributes is relevant.

This library allows combining both. It assumes for any given attribute, there is marginal data available at a high
spatial resolution and a contingency table at lower levels of spatial resolution, and allows combining the two.

With this library, each attribute is added to the population one at a time, conditioned (based on the contingency table)
on previously added attributes but constrained on spatial details (based on the high spatial resolution marginal data).

Note that this library assumes for each attribute, a contingency table is available or can be obtained that exhaustively
lists the possible categorical values the attribute can take. If this table is not available as is, data preparation is
required before this library can be used

## Generating a synthetic population

In the reference implementation, the highest spatial resolution data was available at the level of a neighborhood, so
that example will be maintained here. If only one level of spatial resolution is available, this library provides
less utility, but a column with a single value can be added to the source data to still use this library.

To install this package, call

```bash
pip install gensynthpop @ git+https://github.com/A-Practical-Agent-Programming-Language/GenSynthPop-Python
```

Or add the following line to your projects `requirement.txt`:

```
gensynthpop @ git+https://github.com/A-Practical-Agent-Programming-Language/GenSynthPop-Python
```

## Generate individuals

Start by instantiating a data frame with agent IDs located in each of the neighborhoods:

```python
agent_ids = list()
agent_neighborhoods = list()
agent_count = 0
for neighb_code, (neighb_total) in read_marginal_data(['population'], 'population').iterrows():
    agent_ids += [f"SA{i + agent_count:06d}" for i in range(neighb_total.iloc[0])]
    agent_neighborhoods += [neighb_code] * neighb_total.iloc[0]
    agent_count += neighb_total.iloc[0]
df_synthetic_population = pd.DataFrame(data=dict(agent_id=agent_ids, neighb_code=agent_neighborhoods))
```

Next, add an attribute. For example age group. This is done through the
[Conditional Attribute Adder](gensynthpop/conditional_attribute_adder.py).

Pass in the synthetic population created in the first step, a contingency table that for each age group in the index
specifies the number of people in that age group in each neighborhood.

The `group_by` clause indicates by what column(s) the data is split into high spatial resolution.

```python3
from gensynthpop.conditional_attribute_adder import ConditionalAttributeAdder

df_age_group = pd.read_csv('age_group_marginal_data.csv')
df_synthetic_population = ConditionalAttributeAdder(
        df_synthetic_population=df_synthetic_population,
        df_contingency=df_age_group,
        target_attribute='age_group',
        group_by=['neighb_code']
).run()
```

Next, an attribute can be added conditioned on a previous attribute, provided a contingency table containing at least
that previous attribute is available.

Best results are achieved if the Iterative Proportional Fitting procedure
(e.g. [ipfn-python](https://github.com/AJdeMooij/ipfn/tree/bugfix/pandas-sort-frames)) is applied to the contingency
table first to fit the contingency table to the margins of the attributes already added.

The process is similar to before, but can now specify neighborhood-specific margins by calling `add_margins`. The
arguments work the same as with the
[ipfn-python](https://github.com/AJdeMooij/ipfn/tree/bugfix/pandas-sort-frames) library

```python3
from gensynthpop.conditional_attribute_adder import ConditionalAttributeAdder

df_margins_gender = pd.read_csv('marginal_gender_by_neighborhood.csv')
df_margins_age_group = synthetic_population_to_contingency(
        df_synth_pop, ["neighb_code", "age_group"], True).reset_index()

df_gender_contingency = read_and_fit_gender_contingency_table(df_margins_age_group)

df = ConditionalAttributeAdder(
        df_synthetic_population=df_synthetic_population,
        df_contingency=df_gender_contingency,
        target_attribute="gender",
        group_by=["neighb_code"]
).add_margins(
        margins=[df_margins_age_group, df_margins_gender],
        margins_names=[["age_group"], ["gender"]]
).run()
``````

The process can be repeated for as many attributes as necessary.

## Generate households

After sufficient attributes are added to the individuals, they can be partitioned into households.

The first step is to determine the types of households (e.g., singles, couples without children, couples with 1 child,
couples with 2 children, etc...).

Next, each agent is assigned a *household position* as either an adult or (in the case of households with children)
child in one of those exact households. If a contingency table is available, that is great, but most likely, this
contingency table needs to be constructed. Once it is, add it as an individual-level attribute.

Next, the [Household Grouper](gensynthpop/household_grouper.py) can be used to partition the agents into households
based on their household position. The household grouper aims at placing each agent into a household and generates
households as required. It may switch positions of agents if necessary to fulfill the household constraints.

In order to create the households, three Pandas Series are required:

1) The typical gender distribution between adult partners (`male-female`, `male-male` and `female-female` as index)
2) The typical age disparity between partners of the same gender, where the index defines a range
   as `<range_start>-<range_end>`
3) The typical age disparity between a mother and oldest child, with the range again as index.

Each series should specify the count or relative frequency of each group. If data is not available, a series can be
constructed
with just one record, e.g., `0-999 = 1`, but ideally, these series are constructed from available data for the region of
interest.

Each `HouseholdType` object takes the name of the household and the three distributions as an argument. Next, its
constituent members are specified by household type. A household can have two types of members, `adult` and `child`, and
for each type, the `household position` that was added as an individual level attribute earlier should be specified,
as well as the number of individuals of that type in the household and the household positions from which members can
be taken as a backup.

For example, a household with two parents (with household position `couple_with_1_child`)
and one child (with household position `child_of_couple_with_1_children`) may be specified as follows:

```python3
couple_with_1_child = HouseholdType(
        'couple_with_1_child',
        df_couple_gender_distribution,
        df_couple_age_distribution,
        df_parent_child_age_distribution
).add_members(
        'child_of_couple_with_1_children', 'child', 1, []
).add_members('couple_with_1_child', 'adult', 2, ['couple_no_children', 'single'])
```

If a household has no children, the first call to `add_members` can be omitted. To create single households, the third
argument of the second call to `add_members` can additionally be changed to `1`.

Once all household types are specified, they can be added to the grouper, which can then create the households:

```python3
hh_grouper = HouseholdGrouper(df_synth_pop, ['neighb_code'], 'household_position')
hh_grouper.add_household_type(couple_with_1_child)
df_synthetic_population, df_synthetic_households = hh_grouper.run()
```

## Next steps

When the households have been created, additional attributes can still be added to the synthetic population of
individuals,
or the households can be decorated with additional attributes. The same `ConditionalAttributeAdder` can still be used
for either.
