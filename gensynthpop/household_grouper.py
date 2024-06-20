import math
import numpy as np
import pandas as pd
import warnings
from collections import defaultdict
from typing import List, Literal, Optional, Tuple, Union
from typing_extensions import Self

from gensynthpop.conditional_attribute_adder import _calculate_group_counts

number = Union[int, float]


def calculate_age_range_from_gap(age: Union[int, float], gap_start: Union[int, float], gap_end: Union[int, float]) -> (
        Union[int, float], Union[int, float]):
    age_start = age + gap_start
    age_end = age + gap_end
    if age_start > age_end:
        age_start, age_end = age_end, age_start

    return age_start, age_end


def score_suitability_by_age_disparity(partner_age, age_start, age_end, strict_lower_bound: Optional[int] = None):
    if age_start <= partner_age <= age_end:
        return 0
    elif partner_age < age_start:
        diff = age_start - partner_age
        if strict_lower_bound is not None and partner_age < strict_lower_bound:
            diff += 999
        return diff
    else:
        assert partner_age > age_end
        return partner_age - age_end


def score_sibling_age_suitability(age: int, *reference_ages: int):
    if age in reference_ages:
        return 10
    else:
        return min(abs(age - r) for r in reference_ages)


def find_sibling_from_pool(pool: pd.DataFrame, msk: pd.Series, sibling_ages):
    candidates = pool.loc[msk][["age"]]
    candidates.loc[:, 'suitability'] = candidates.age.apply(
            lambda x: score_sibling_age_suitability(x, *sibling_ages))
    candidates = candidates.sort_values(by=['suitability'], ascending=True)
    sibling = candidates.iloc[0]
    return sibling


class HouseholdType:
    df_synth_pop: pd.DataFrame
    household_position_column: str

    def __init__(self, household_type: str, couple_gender_distribution: pd.Series,
                 couple_age_distribution: pd.Series, parent_child_age_distribution: pd.Series):
        self.hh_type = household_type
        self.positions = list()
        self.position_identifiers = dict()
        self.households = dict()
        self.couple_gender_distribution = couple_gender_distribution
        self.couple_age_distribution = couple_age_distribution
        self.parent_child_age_distribution = parent_child_age_distribution
        self.sampled_agents = []

    def add_members(
            self,
            household_position: str,
            position_identifier: Literal['adult', 'child'],
            amount: int,
            backup_position_identifiers: List[str]
    ):
        self.positions.append(dict(
                position_identifier=position_identifier,
                position=household_position if isinstance(household_position, list) else [household_position],
                amount=amount,
                backup_position_identifiers=backup_position_identifiers
        ))
        self.position_identifiers[position_identifier] = len(self.positions) - 1
        return self

    def update_state(self, df_synth_pop: pd.DataFrame, household_position_column: str) -> None:
        self.df_synth_pop = df_synth_pop
        self.household_position_column = household_position_column

    def get_position_for_name(self, position: str) -> dict:
        return self.positions[self.position_identifiers[position]]

    def agent_to_household(self) -> pd.DataFrame:
        """
        Adds the household ID of the household agents have been assigned to, to the synthetic population
        Returns:

        """
        households = pd.Series({agent: household_id for household_id in self.households.keys() for agent in
                                self.households[household_id]['all']})

        self.df_synth_pop.loc[households.index, 'household_id'] = households

        return self.df_synth_pop

    def households_to_dataframe(self):
        """
        Constructs a household data frame based on the current household partitioning
        Returns:

        """
        ids = list()
        hh_size = list()
        neighb_code = list()
        hh_type = list()
        for id, hh in self.households.items():
            ids.append(id)
            hh_size.append(len(hh['all']))
            neighb_code.append(self.df_synth_pop.loc[hh['all'][0], 'neighb_code'])
            hh_type.append(self.hh_type)

        households = pd.DataFrame(dict(
                neighb_code=neighb_code,
                hh_type=hh_type,
                hh_size=hh_size
        ), index=ids)
        households.index.name = 'household_id'

        return households

    def get_all_agents(self):
        return [a for h in self.households.values() for a in h['all']]

    def check_integrity(self) -> bool:
        """
        Verifies all agents in the current group have been assigned a household, and that no agent is assigned
        more than one household
        Returns:
            True iff integrity is maintained
        """
        all_agents = self.get_all_agents()
        u, i = np.unique(all_agents, return_inverse=True)
        duplicate = u[np.bincount(i) > 1]
        assert len(duplicate) == 0, print(duplicate)

        all_positions = [hh_position for p in self.positions for hh_position in p['position']]
        msk = self.df_synth_pop[self.household_position_column].isin(all_positions)
        aids = self.df_synth_pop[msk].index.values

        missing = list()
        for aid in aids:
            if aid not in all_agents:
                missing.append(aid)

        if len(missing):
            warnings.warn(
                    f"Failed to assign {len(missing)} agents into a household for {self.hh_type}. {msk.sum()} agents "
                    f"in mask. Total assigned agents is {len(all_agents)}.")
            print(self.df_synth_pop.loc[missing][["neighb_code", "age", "gender", self.household_position_column]])
        assert len(missing) == 0, len(missing)

        return True

    def create_household_with_id(self, position: dict, id_offset: int, agents: list[str]):
        self.households[f"SSH{id_offset:06d}"] = {
            position['position_identifier']: agents,
            'all': agents
        }

    def mask_with_remaining_agents(self, df, msk):
        msk &= ~df.index.isin(self.sampled_agents)
        return df[msk]

    def get_remaining_agents_in_position(self, position: Union[str, list[str]],
                                         msk: Optional = None):
        if isinstance(position, str):
            position = [position]
        position_msk = self.df_synth_pop[self.household_position_column].isin(position)
        if msk is None:
            msk = position_msk
        else:
            msk = msk & position_msk

        return self.mask_with_remaining_agents(self.df_synth_pop, msk)

    def get_base_child_mask(self) -> np.ndarray[bool]:
        child_position = self.get_position_for_name('child')
        msk_children = self.df_synth_pop[self.household_position_column].isin(child_position['position'])
        return msk_children

    def get_base_adult_mask(self) -> np.ndarray[bool]:
        parent_position = self.get_position_for_name('adult')
        msk_parent = self.df_synth_pop[self.household_position_column].isin(parent_position['position'])
        return msk_parent

    def create_from_members(self, msk: np.ndarray[bool], id_offset: int) -> int:
        if self.get_position_for_name('adult')['amount'] == 2:
            parents = self.pair_partners(msk)
        else:
            parents = self.create_singles(msk)

        if 'child' in self.position_identifiers:
            child_position = self.get_position_for_name('child')
            children = self.group_children(msk, child_position)
            print(len(children), "sets of children vs", len(parents), "sets of parents")
            id_offset = self.match_adults_with_children(parents, children, id_offset)
            print("Created", len(self.households), "households")
        else:
            print("Creating households without children")
            for parent in parents:
                self.create_household_with_id(self.get_position_for_name('adult'), id_offset, [p[0] for p in parent])
                print(f"Created HH with {parent}")
                id_offset += 1

        return id_offset

    def match_adults_with_children(self, parents: list[list[tuple[str, int, str]]], children: list[list[str]],
                                   id_offset: int) -> int:
        """
        Attempts to match adult-clusters with child-clusters.
        The current implementation supports only comparing by age.
        Args:
            parents:
            children:
            id_offset:

        Returns:
            Updated ID offset
        """
        age_mother = _calculate_group_counts(self.parent_child_age_distribution, len(children))

        parent_ages = list()
        if self.get_position_for_name('adult')['amount'] == 2:
            for p1, p2 in parents:
                if p1[2] == p2[2]:
                    parent_ages.append(min(p1[1], p2[1]))
                else:
                    parent_ages.append(p1[2] if p1[2] == 'female' else p2[1])
        else:
            parent_ages = [p[0][1] for p in parents]

        parent_ages = pd.DataFrame(dict(age=parent_ages))
        parent_msk = parent_ages.age.transform(lambda _: True)

        children_with_ages = sorted([(child, self.df_synth_pop.loc[child].age.max()) for child in children],
                                    key=lambda x: x[1], reverse=True)

        child_offset = 0
        # for age_gap, count in age_mother.items():
        for age_gap, count in sorted(age_mother.items()):
            count = age_mother[age_gap]
            gap_start, gap_end = map(int, str(age_gap).split("-"))
            for _ in range(count):
                siblings, child_age = children_with_ages[child_offset]

                parent_candidates = parent_ages.loc[parent_msk, ['age']]
                parent_candidates.loc[:, 'suitability'] = parent_candidates.age.transform(
                        lambda x: score_suitability_by_age_disparity(x, gap_start + child_age, gap_end + child_age,
                                                                     strict_lower_bound=child_age + 14)
                )
                parent_candidates = parent_candidates.sort_values(by='suitability', ascending=True)
                parent_idx = parent_candidates.iloc[0].name
                parent_msk &= parent_ages.index != parent_idx
                parent = parents[parent_idx]
                parent_age = parent_ages.loc[parent_idx].values[0]
                if parent_age - child_age < 14:
                    raise ValueError(
                            f"Child age gap too small: {parent_age} - {child_age} = {parent_age - child_age}")
                household = [p[0] for p in parent] + [c for c in siblings]
                self.create_household_with_id(self.get_position_for_name('child'), id_offset, household)
                print(
                        f"{parent_age=}, {child_age=} Matched parent {parent} to "
                        f"{children[child_offset]} "
                        f"with (oldest) age {child_age}")
                id_offset += 1
                child_offset += 1

        if parent_msk.sum() > 0:
            warnings.warn(f"Creating {parent_msk.sum()} couples without children")
            for parent_idx in parent_ages.loc[parent_msk].index:
                parent = parents[parent_idx]
                self.create_household_with_id(self.get_position_for_name('adult'), id_offset,
                                              [p[0] for p in parent])
                print(f"Created HH with {parent}")
                id_offset += 1

        return id_offset

    def create_singles(self, group_msk):
        msk = group_msk & self.get_base_adult_mask()
        parent_position = self.get_position_for_name('adult')
        n_couples = math.ceil(msk.sum() / parent_position['amount'])
        if 'child' in self.position_identifiers:
            n_children = math.ceil(
                    (group_msk & self.get_base_child_mask()).sum() / self.get_position_for_name('child')['amount'])
            n_couples = max(n_couples, n_children)

        single_households = list()

        for i in range(n_couples):
            p = self.find_primary_partner(group_msk, parent_position['position'],
                                          parent_position['backup_position_identifiers'])
            single_households.append([(p.name, p.age, p.gender)])
            self.sampled_agents.append(p.name)

        return single_households

    def pair_partners(self, group_msk) -> (List[List[Tuple[str, int, str]]]):
        msk = group_msk & self.get_base_adult_mask()
        parent_position = self.get_position_for_name('adult')
        n_couples = math.ceil(msk.sum() / parent_position['amount'])
        if 'child' in self.position_identifiers:
            n_children = math.ceil(
                    (group_msk & self.get_base_child_mask()).sum() / self.get_position_for_name('child')['amount'])
            n_couples = max(n_couples, n_children)

        couple_gender_count = _calculate_group_counts(self.couple_gender_distribution, n_couples)

        couples = list()

        for (first_partner, second_partner), count in couple_gender_count.items():
            print(f"Creating {count} {first_partner}-{second_partner} couples")
            if first_partner == second_partner:
                for i in range(count):
                    p1 = self.find_primary_partner(group_msk, parent_position['position'],
                                                   parent_position['backup_position_identifiers'],
                                                   first_partner)
                    self.sampled_agents.append(p1.name)
                    p2 = self.find_primary_partner(group_msk, parent_position['position'],
                                                   parent_position['backup_position_identifiers'],
                                                   first_partner)
                    self.sampled_agents.append(p2.name)
                    couples.append(
                            [(p1.name, p1.age, p1.gender), (p2.name, p2.age, p2.gender)])
            else:
                couple_age_count = _calculate_group_counts(self.couple_age_distribution, count)
                for age_gap, age_cap_count in couple_age_count.items():
                    gap_start, gap_end = map(int, str(age_gap).rsplit("-", 1))
                    gap_end = math.copysign(gap_end, gap_start)

                    for i in range(age_cap_count):
                        primary_partner = self.find_primary_partner(
                                group_msk,
                                parent_position['position'],
                                parent_position['backup_position_identifiers'],
                                first_partner)

                        self.sampled_agents.append(primary_partner.name)
                        secondary_partner = self.find_secondary_partner(
                                group_msk,
                                primary_partner, parent_position['position'],
                                parent_position['backup_position_identifiers'],
                                gap_start, gap_end, second_partner)
                        if secondary_partner is not None:
                            self.sampled_agents.append(secondary_partner.name)
                            couples.append([(primary_partner.name, primary_partner.age, primary_partner.gender),
                                            (secondary_partner.name, secondary_partner.age, secondary_partner.gender)])

                        else:
                            warnings.warn("Only single parent found")
                            couples.append([primary_partner.name, primary_partner.age, primary_partner.gender])

        return couples

    def find_primary_partner(self, msk: np.ndarray[bool], primary_position_value: str,
                             backup_position_values: List[str], gender: Optional[str] = None):
        candidates = self.get_remaining_agents_in_position(
                primary_position_value,
                msk)
        correct_gender_candidates = candidates if gender is None else candidates[candidates.gender == gender]
        if len(correct_gender_candidates) > 0:
            return correct_gender_candidates.iloc[0]
        else:
            if len(candidates):
                wrong_candidate = candidates.iloc[0]
                for position in backup_position_values:
                    c = self.find_opposite_gender_replacement_for_candidate(wrong_candidate, msk, position)
                    if c is not None:
                        self.switch_household_positions(wrong_candidate.name, c)
                        primary_partner = self.df_synth_pop.loc[c]
                        assert primary_partner[
                                   self.household_position_column] in primary_position_value, (
                            f"Got {primary_partner[self.household_position_column]}, expected one of "
                            f"{primary_position_value}")
                        return primary_partner
                print(
                        f"Could not find replacement with {gender} in backup positions. Returning wrong candidate from "
                        f"current pool")
                return wrong_candidate
            else:
                for position in backup_position_values:
                    candidates = self.get_remaining_agents_in_position(
                            position, msk & (self.df_synth_pop.gender == gender))
                    if len(candidates) > 0:
                        candidate = candidates.iloc[0]
                        print(
                                f"Moving {candidate.name} from {candidate[self.household_position_column]} to "
                                f"{primary_position_value[0]}")
                        self.df_synth_pop.loc[candidate.name, self.household_position_column] = primary_position_value[
                            0]
                        return candidates.iloc[0]

                print("Still no candidate found. Trying to find candidate with wrong gender in backup pool")
                for position in backup_position_values:
                    candidates = self.get_remaining_agents_in_position(position, msk)
                    if len(candidates) > 0:
                        candidate = candidates.iloc[0]
                        print(f"Moving {candidate.name} from {candidate[self.household_position_column]} to "
                              f"{primary_position_value[0]}")
                        self.df_synth_pop.loc[candidate.name, self.household_position_column] = primary_position_value[
                            0]
                        return candidate

            raise ValueError(f"Unable to find primary partner in backup positions {backup_position_values}")

    def find_secondary_partner(
            self,
            msk: np.ndarray[bool],
            primary_partner,
            primary_position_value,
            backup_position_values,
            gap_start, gap_end,
            gender: str
    ):
        candidates = self.find_couple_candidates(msk, primary_position_value, primary_partner, gap_start, gap_end)
        other_gender_candidates = candidates[candidates.gender == gender]

        if not len(other_gender_candidates):
            if len(candidates):
                wrong_candidate = candidates.iloc[0]

                for position in backup_position_values:
                    c = self.find_opposite_gender_replacement_for_candidate(wrong_candidate, msk, position)

                    if c is not None:
                        self.switch_household_positions(wrong_candidate.name, c)
                        secondary_partner = self.df_synth_pop.loc[c]
                        assert secondary_partner[self.household_position_column] in primary_position_value, (
                            f"Got {secondary_partner[self.household_position_column]}, expected on of"
                            f"{primary_position_value}")
                        break
                else:
                    # Tough luck, even in backup no candidates
                    print(
                            f"Couldn't find secondary for {primary_partner.name}. Pairing to {wrong_candidate.name} of "
                            f"same gender")
                    secondary_partner = wrong_candidate
            else:
                for position in backup_position_values:
                    candidates = self.find_couple_candidates(msk, [position], primary_partner, gap_start, gap_end)
                    candidates = candidates[candidates.gender == gender]
                    if len(candidates):
                        secondary_partner = candidates.iloc[0]
                        self.df_synth_pop.loc[
                            secondary_partner.name, self.household_position_column] = primary_position_value[0]
                        print(
                                f"Found {secondary_partner.name} as best fit for {primary_partner.name} from "
                                f"{position} with opposite gender. Moving from {position} to "
                                f"{primary_position_value[0]}")

                        break
                else:
                    # Try same trick, but don't care about gender
                    for position in primary_position_value + backup_position_values:
                        candidates = self.find_couple_candidates(msk, [position], primary_partner, gap_start, gap_end)
                        if len(candidates):
                            secondary_partner = candidates.iloc[0]
                            self.df_synth_pop.loc[
                                secondary_partner.name, self.household_position_column] = primary_position_value[0]
                            print(
                                    f"Found {secondary_partner.name} as best partner for {primary_partner.name} from "
                                    f"{position}, but with same gender. "
                                    f"Moving from {position} to {primary_position_value[0]}")
                            break
                    else:
                        print(f"No more suitable candidates. {primary_partner.name} will have to be single")
                        return None

        else:
            secondary_partner = other_gender_candidates.iloc[0]

        return secondary_partner

    def switch_household_positions(self, agent_1: str, agent_2: str):
        """
        In case a more suitable candidate from one of the backup positions is found, the household position of the
        two candidates is switched
        Args:
            agent_1:
            agent_2:

        Returns:

        """
        a1 = self.df_synth_pop.loc[agent_1]
        a2 = self.df_synth_pop.loc[agent_2]

        assert a1.neighb_code == a2.neighb_code

        print(
                f"Moving {a1.name} ({a1.age} {a1.gender} {a1[self.household_position_column]}) to "
                f"{a2[self.household_position_column]}"
                f" and {a2.name} ({a2.age} {a2.gender} {a2[self.household_position_column]}) back to "
                f"{a1[self.household_position_column]} as replacement"
        )

        a1_pos = self.df_synth_pop.loc[agent_1, self.household_position_column]
        a2_pos = self.df_synth_pop.loc[agent_2, self.household_position_column]

        self.df_synth_pop.loc[agent_1, self.household_position_column] = a2_pos
        self.df_synth_pop.loc[agent_2, self.household_position_column] = a1_pos

    def find_couple_candidates(self, msk: np.ndarray[bool], position_value: list[str], primary_partner,
                               gap_start: number, gap_end: number):
        """
        Attempts to find a suitable partner for the primary partner passed, based on the start and end of the
        age gap currently being considered

        Args:
            msk:
            position_value:
            primary_partner:
            gap_start:
            gap_end:

        Returns:

        """
        msk = msk & self.df_synth_pop[self.household_position_column].isin(position_value)
        msk &= ~self.df_synth_pop.index.isin(self.sampled_agents)

        candidates = self.df_synth_pop[msk].copy()
        age_start, age_end = calculate_age_range_from_gap(primary_partner.age, gap_start, gap_end)
        candidates.loc[:, 'suitability'] = candidates.age.transform(
                lambda x: score_suitability_by_age_disparity(x, age_start, age_end)
        )
        return candidates.sort_values(by='suitability', ascending=True)

    def find_opposite_gender_replacement_for_candidate(self, wrong_candidate, msk, position: str) -> Optional:
        """
        When the balance of male/female agents is insufficient to satisfy the number of different sex couples, a
        candidate can be switched out with another candidate from the backup pool that is similar to the candidate
        from the current pool, except with opposite gender.
        Args:
            wrong_candidate:
            msk:
            position:

        Returns:

        """
        msk = msk & (self.df_synth_pop[self.household_position_column] == position)
        msk &= self.df_synth_pop.gender != wrong_candidate.gender
        msk &= ~self.df_synth_pop.index.isin(self.sampled_agents)
        df = self.df_synth_pop[msk].drop('gender', axis=1)

        if len(df) > 0:
            similarity_scores = df.apply(lambda row: sum([row[c] == wrong_candidate[c] for c in row.index]), axis=1)
            return similarity_scores.idxmax()

        return None

    def group_children(self, msk, child_position: dict) -> List[List[str]]:
        position, n_children = child_position['position'], child_position['amount']
        print(f"Grouping children into sets of {n_children}")
        households = list()
        pool = self.df_synth_pop[(self.df_synth_pop[self.household_position_column].isin(position)) & msk]
        pool = pool.sample(frac=1)  # To avoid first sampling only one gender and then all the others
        pool_mask = pool.age.transform(lambda _: True)
        while pool_mask.sum() > 0:
            first_child = pool.loc[pool_mask].iloc[0]
            self.sampled_agents.append(first_child.name)
            children = [first_child.name]
            sibling_ages = [first_child.age]
            pool_mask &= (pool.index != first_child.name)
            for _ in range(n_children - 1):
                if pool_mask.sum() == 0:
                    # Can't fulfill this household
                    break
                sibling = find_sibling_from_pool(pool, pool_mask, sibling_ages)
                children.append(sibling.name)
                sibling_ages.append(sibling.age)
                pool_mask &= (pool.index != sibling.name)
                self.sampled_agents.append(sibling.name)

            households.append(children)

        check_msk = msk & (self.df_synth_pop[self.household_position_column].isin(position)) & (
            ~self.df_synth_pop.index.isin(self.sampled_agents))
        assert check_msk.sum() == 0, str(check_msk.sum()) + " " + str(pool_mask.sum())

        return households


class HouseholdGrouper:

    def __init__(
            self,
            df_synth_pop: pd.DataFrame,
            group_by: Optional[List[str]],
            household_position_column: str = 'household_position',
    ):
        self.df_synth_pop = df_synth_pop.set_index('agent_id')
        self.df_synth_pop.loc[:, 'household_id'] = None
        self.group_by = group_by
        self.target_column = household_position_column
        self.household_types = list()

    def add_household_type(self, household_type: HouseholdType) -> Self:
        self.household_types.append(household_type)
        return self

    def run(self):
        offset = 0
        for household_type in self.household_types:
            household_type.update_state(self.df_synth_pop, self.target_column)
            for group_name, _ in self.df_synth_pop.groupby(self.group_by):
                msk = (self.df_synth_pop[self.group_by] == group_name).all(axis=1)
                print(f"Distributing {msk.sum()} agents into {household_type.hh_type} for {group_name}")
                offset = household_type.create_from_members(msk, offset)

            if household_type.check_integrity():
                household_type.agent_to_household()

        synthetic_households = pd.concat([ht.households_to_dataframe() for ht in self.household_types])

        agents_without_household_msk = self.df_synth_pop.household_id.isna()
        if agents_without_household_msk.sum() > 0:
            msg = f"{agents_without_household_msk.sum()} agents without household ID\n"
            missing_groups = [(n, g.index) for n, g in
                              self.df_synth_pop[agents_without_household_msk].groupby(
                                      self.group_by + [self.target_column])]

            for ((g, position), g) in missing_groups:
                msg += f"\t {g} \t {position}: \t\t {len(g)} missing: {g}\n"
            warnings.warn(msg)
        else:
            print(f"All agents have a household type")

        all_agents_dct = defaultdict(list)
        all_agents = list()
        for household_type in self.household_types:
            hh_agens = household_type.get_all_agents()
            all_agents += hh_agens
            for a in hh_agens:
                all_agents_dct[a].append(household_type.hh_type)

        u, i = np.unique(all_agents, return_inverse=True)
        duplicate = u[np.bincount(i) > 1]
        if len(duplicate):
            warnings.warn("The following agents were assigned in multiple household types:")
            for a in duplicate:
                print("\t", a, "\t", all_agents_dct[a])
        else:
            print("Each agent appears in at most one household only")

        msk_not_assigned = self.df_synth_pop.household_id.isna()
        msk_assigned = ~msk_not_assigned

        if msk_not_assigned.sum() > 0:
            warnings.warn(f"{msk_not_assigned.sum()} agents were not assigned to any household")
        print(f"Households assigned to {msk_assigned.sum()} of {(msk_assigned | msk_not_assigned).sum()} agents")

        # Show resulting DF of all agents with household
        print(self.df_synth_pop[msk_assigned])

        return self.df_synth_pop, synthetic_households
