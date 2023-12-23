# Author: Mahir Salihbasic (mahir.salihbasic@untz.ba)
#
# Gale-Shapley stable marriage algorithm implementation with numpy.

from enum import Enum
from typing import Union, Tuple

import numpy as np
import numpy.typing as nptyp

def gen_sample_sm_sample(sample_size: int) -> nptyp.NDArray[np.int32]:
    """
    Randomly generate a square matrix of preferences.

    Preferences are represented by a matrix, one for males and the other for females, 
    such that each row corresponds to a specific male (or female) and each column to a specific
    female (or male) that the male (or female) ranks.

    In other words, if we have a male preference matrix M_ij, then M(i,j) should give us the
    i-th male's preference for j-th female.
    Similarly, if we have a female preference matrix F_ij, then F(i,j) shoul give us the i-th female's
    preference for the j-th male.
    """
    if sample_size <= 1:
        raise ValueError("Sample size must be greater than 1.")

    rng = np.random.default_rng()

    sample = np.empty((sample_size, sample_size), dtype=np.int32)

    for i in range(0, sample_size):
        s = rng.choice(sample_size, sample_size, replace=False)
        sample[i] = s

    return sample

# Once we have our preferences array, we can create a class with useful helper methods
# on our preferences
class Preferences():
    """
    Wrapper class around two specific male and female preferences matrices.

    Its purpose is to provide methods that allow connection between the two, such as 
    preference comparisons and modification.
    """

    def __init__(self, m_sample: nptyp.NDArray[np.int32], f_sample: nptyp.NDArray[np.int32], sample_size: int) -> None:
        if sample_size <= 1:
            raise ValueError("Sample size must be greater than 1.")
        
        if m_sample.shape != (sample_size, sample_size):
            raise ValueError("Passed inappropriate matrix of male preferences! Expected square matrix.")

        if f_sample.shape != (sample_size, sample_size):
            raise ValueError("Passed inappropriate matrix of female preferences! Expected square matrix.")

        if m_sample.shape != f_sample.shape:
            raise ValueError("Male and female sample matrices must have the same shape!")
        
        self.m_sample = m_sample
        self.f_sample = f_sample
        self.sample_size = sample_size
    
    def male_ranking_of_females(self, m_id: int) -> nptyp.NDArray[np.int32]:
        "Returns an array (from 0 to f_id - 1) of females that the specified m_id ranks."

        return self.m_sample[m_id,:]
    
    def male_ranking_of_specific_female(self, m_id: int, f_id: int) -> nptyp.NDArray:
        "Returns the ranking of the specified f_id by the specified m_id."

        return self.m_sample[m_id, f_id]
    
    def highest_ranked_female_by_male(self, m_id: int) -> int:
        "Returns the index of the highest ranked female by the provided male."
        return self.m_sample.argmax(axis=1)[m_id]
    
    def male_strike_specific_female(self, m_id: int, f_id: int) -> None:
        "Has m_id strike (i.e set to -1) preference of f_id."

        self.m_sample[m_id, f_id] = -1
    
    def has_male_striked_specific_female(self, m_id: int, f_id: int) -> bool:
        "Checks if the m_id striked the provided f_id."

        return self.male_ranking_of_specific_female(m_id, f_id) == -1

    def female_ranking_of_males(self, f_id: int) -> nptyp.NDArray:
        "Returns an array (from 0 to m_id - 1) of males that the specified f_id ranks."

        return self.f_sample[f_id,:]
    
    def female_ranking_of_specific_male(self, f_id: int, m_id: int) -> nptyp.NDArray:
        "Returns the ranking of the specified m_id by the specified f_id."

        return self.f_sample[f_id, m_id]
    
    def female_prefers_first_to_second(self, f_id: int, m_id1: int, m_id2: int) -> bool:
        "Whether the specified f_id prefers m_id1 to m_id2."

        return self.female_ranking_of_specific_male(f_id, m_id1) > self.female_ranking_of_specific_male(f_id, m_id2)

class MatchingResult(Enum):
    """
    When attempting to match a male to a female, three things can happen:

        MATCHED_BY_DEFAULT - the female was not matched to anyone, therefore she is automatically matched to this male

        REJECTED - the female was matched to a better (preferred) male, therefore the current male is rejected

        MATCHED_OVER_PREVIOUS - the female rejects the previous matches and matches to the new male (preferred to previous)
    """
    REJECTED = 0
    MATCHED_BY_DEFAULT = 1
    MATCHED_OVER_PREVIOUS = 2

class Matcher():
    """
    The Matcher class is responsible for keeping track of all matched and unmatched
    males and females in the given preferences sample.

    All the matches are kept in a dictionary (named 'matches'), where the key is an f_id and the value
    is the m_id to which the female is matched.

    Whether the specified man is matched is kept in a specific array (named 'matched_men') where
    the array index corresponds to the specific m_id. If the value at specific index is 0, that m_id
    is considered unmatched, otherwise the m_id is considered matched.

    The method 'attempt_match' which takes an m_id and an f_id attempts to match a specific male to a specific
    female according to the Gale-Shapely algorithm that this package implements.
    """

    def __init__(self, preferences: Preferences) -> None:
        self.preferences = preferences
        self.matches: dict[np.int32, np.int32] = {}
        self.matched_men: nptyp.NDArray[np.int32] = np.zeros(dtype=np.int32, shape=self.preferences.sample_size)
        self.matched_count: int = 0

    def get_matches(self) -> dict[np.int32, np.int32]:
        "Returns the current matches."

        return self.matches

    def is_matched(self, m_id: int) -> bool:
        "Returns true if the specified m_id is matched to some f_id."
        return self.matched_men[m_id] != 0
    
    def _set_matched(self, m_id: int) -> None:
        "Declares the specified m_id as matched and raises the total count of matched men by 1."

        self.matched_men[m_id] = 1
        self.matched_count += 1

    def _set_unmatched(self, m_id: int) -> None:
        "Declares the specfiied m_id as unmatched and decreases the total count of matched men by 1."

        self.matched_men[m_id] = 0
        self.matched_count -= 1

    def attempt_match(self, m_id: int, f_id: int) -> MatchingResult:
        """
        Attempt to match the specified unmatched m_id with the specified f_id. 

        If f_id is not already matched, finish matching.
        If f_id is already matched, check the preferences array to see if f_id prefers
        m_id to whomever f_id is already matched, if yes, then remove the old m_id from 
        matched list and match f_id with the new m_id, otherwise do nothing.
        
        Raises a ValueError if the provided m_id is already matched.

        Returns the appropriate MatchingResult.
        """

        if self.is_matched(m_id):
            raise ValueError("This m_id is already matched!")
        
        if f_id not in self.matches.keys():
            self.matches[f_id] = m_id
            self._set_matched(m_id)
            return MatchingResult.MATCHED_BY_DEFAULT
        else:
            current_m_id = self.matches.get(f_id)
            if self.preferences.female_prefers_first_to_second(f_id, m_id, current_m_id):
                # Match the new male
                self.matches[f_id] = m_id
                self._set_matched(m_id)

                # Unmatch the old male and strike this female for him
                # (Yes, this sounds bad... I am well aware... Sorry...)
                self._set_unmatched(current_m_id)
                self.preferences.male_strike_specific_female(current_m_id, f_id)

                return MatchingResult.MATCHED_OVER_PREVIOUS
            else:
                self.preferences.male_strike_specific_female(m_id, f_id)
                return MatchingResult.REJECTED


class MarriageArranger():
    """
    This class serves as a thin wrapper around the Matcher class in order to provide
    the basic functionality of a stable marriage problem solver.
    """

    def __init__(self, preferences: Preferences):
        self.preferences = preferences
        self.matcher = Matcher(self.preferences)
        self.finished: bool = False
    
    def pass_day_for_man(self, m_id: int) -> Union[MatchingResult, None]:
        if self.matcher.is_matched(m_id):
            return None
        
        highest_ranked_f_idx = self.preferences.highest_ranked_female_by_male(m_id)
        return self.matcher.attempt_match(m_id, highest_ranked_f_idx)

    def pass_day(self) -> Tuple[int, int]:

        new_matches: int = 0
        new_unmatches: int = 0

        for m_id in range(0, self.preferences.m_sample.shape[0]):
            result: Union[MatchingResult, None] = self.pass_day_for_man(m_id)

            match result:
                case None: continue
                case MatchingResult.REJECTED: continue
                case MatchingResult.MATCHED_BY_DEFAULT:
                    new_matches += 1
                    continue
                case MatchingResult.MATCHED_OVER_PREVIOUS:
                    new_matches += 1
                    new_unmatches += 1
                    continue
        
        return (new_matches, new_unmatches)
    
    def pass_days(self) -> Union[dict[np.int32, np.int32], int]:
        """
        Returns the final state of all matches together with the number of days
        it took to reach it.
        """

        current_day: int = 1

        while not self.finished:
            self.pass_day()
            current_day += 1
            if len(self.matcher.matches.keys()) == self.preferences.sample_size:
                self.finished = True

        return (self.matcher.get_matches(), current_day)