from collections import Counter, defaultdict
from typing import List
from math_verify import verify

MAJORITY_VOTE = "majority_vote"
ORM_VOTE = "orm_vote"
ORM_MAX = "orm_max"
PRM_MIN_MAX = "prm_min_max"
PRM_MIN_VOTE = "prm_min_vote"
PRM_LAST_MAX = "prm_last_max"
PRM_LAST_VOTE = "prm_last_vote"


def _agg_majority_vote(x_list: List[str], unused_v_list: List[List[float]]):
    groups = []

    for x in x_list:
        found = False
        for i in range(len(groups)):
            representative, count = groups[i]
            if verify(representative, x):
                groups[i] = (representative, count + 1)
                found = True
                break
        if not found:
            groups.append((x, 1))

    max_count = -1
    result = None
    for representative, count in groups:
        if count > max_count:
            max_count = count
            result = representative
    return result


def _agg_orm_vote(x_list: List[str], v_list: List[float]) -> str:
    assert len(x_list) == len(v_list)
    groups = []

    for x, v in zip(x_list, v_list):
        found = False
        for i in range(len(groups)):
            representative, total = groups[i]
            if verify(representative, x):
                groups[i] = (representative, total + v)
                found = True
                break
        if not found:
            groups.append((x, v))

    max_total = -float("inf")
    result = None
    for representative, total in groups:
        if total > max_total:
            max_total = total
            result = representative
    return result


def _agg_orm_max(x_list: List[str], v_list: List[float]):
    text_max = x_list[v_list.index(max(v_list))]
    return text_max


def _agg_prm_min_max(x_list: List[str], v_list: List[List[float]]):
    v_list = [min(v) if v else -1.0 for v in v_list]
    return _agg_orm_max(x_list, v_list)


def _agg_prm_last_max(x_list: List[str], v_list: List[List[float]]):
    v_list = [v[-1] if v else -1.0 for v in v_list]
    return _agg_orm_max(x_list, v_list)


def _agg_prm_min_vote(x_list: List[str], v_list: List[List[float]]):
    v_list = [min(v) if v else -1.0 for v in v_list]
    return _agg_orm_vote(x_list, v_list)


def _agg_prm_last_vote(x_list: List[str], v_list: List[List[float]]):
    v_list = [v[-1] if v else -1.0 for v in v_list]
    return _agg_orm_vote(x_list, v_list)


AGG_FN_MAP = {
    MAJORITY_VOTE: _agg_majority_vote,
    # ORM_VOTE: _agg_orm_vote,
    # ORM_MAX: _agg_orm_max,
    PRM_MIN_MAX: _agg_prm_min_max,
    PRM_MIN_VOTE: _agg_prm_min_vote,
    PRM_LAST_MAX: _agg_prm_last_max,
    PRM_LAST_VOTE: _agg_prm_last_vote,
}
