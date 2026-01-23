"""Task 3: State-aware dynamic elevator parking policy.

Simple, interpretable rule-based mapping from (state, y_hat) -> desired parking allocation
across zones L = {Lobby, Mid, Upper} and a minimal repositioning plan for a fleet of
N elevators.

Public functions:
- parking_policy(state: str, y_hat: float, n_elevators: int = 4, current_alloc: dict | None = None)
    -> returns dict with desired_alloc, moves (list of (from_idx, to_zone)), and metadata.

- compute_desired_allocation(state, y_hat, n_elevators)
    -> deterministic allocation counts per zone (sums to n_elevators)

- compute_reposition_moves(current_alloc_list, desired_alloc)
    -> minimal moves to reach desired allocation (greedy)

The design keeps behavior simple so it is easy to test and reason about. This module
is deliberately decoupled from a motion simulator; it only returns planned moves.
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any

ZONES = ["Lobby", "Mid", "Upper"]


def _proportions_from_state(state: str, y_hat: float) -> Dict[str, float]:
    """Return target proportions for (Lobby, Mid, Upper) given a traffic state and intensity.

    y_hat is used to slightly bias allocation when demand is very low or very high.
    """
    # Base templates (interpretable): values sum to 1
    templates = {
        'Morning Up-Peak': {'Lobby': 0.7, 'Mid': 0.25, 'Upper': 0.05},
        'Lunch Down-Peak': {'Lobby': 0.2, 'Mid': 0.6, 'Upper': 0.2},
        'Evening Down-Peak': {'Lobby': 0.05, 'Mid': 0.25, 'Upper': 0.7},
        'Afternoon Mixed': {'Lobby': 0.33, 'Mid': 0.34, 'Upper': 0.33},
        'Night Idle': {'Lobby': 0.6, 'Mid': 0.2, 'Upper': 0.2},
        'Weekend Low-Demand': {'Lobby': 0.5, 'Mid': 0.3, 'Upper': 0.2},
    }

    tpl = templates.get(state, templates['Afternoon Mixed'])

    # Soft adjustment by intensity: if y_hat is very small (<1), bias toward idle (Lobby);
    # if very large, concentrate a bit more where state suggests.
    if y_hat <= 1:
        adj = {z: (tpl[z] * 0.4 if z != 'Lobby' else tpl[z] + 0.4) for z in ZONES}
    elif y_hat >= 10:
        # amplify primary zone by 10% of its share
        primary = max(tpl, key=tpl.get)
        adj = {z: (tpl[z] + 0.10 * tpl[primary] if z == primary else tpl[z] - 0.05 * tpl[primary]) for z in ZONES}
    else:
        adj = tpl

    # normalize
    total = sum(adj.values())
    return {z: float(adj[z] / total) for z in ZONES}


def compute_desired_allocation(state: str, y_hat: float, n_elevators: int) -> Dict[str, int]:
    """Compute integer desired counts per zone summing to n_elevators.

    Uses rounding with a deterministic tie-break to ensure sum matches.
    """
    props = _proportions_from_state(state, y_hat)
    raw = {z: props[z] * n_elevators for z in ZONES}
    floored = {z: int(raw[z]) for z in ZONES}
    remaining = n_elevators - sum(floored.values())

    # Distribute remaining by largest fractional parts
    fracs = sorted([(z, raw[z] - floored[z]) for z in ZONES], key=lambda x: (-x[1], ZONES.index(x[0])))
    for i in range(remaining):
        z = fracs[i][0]
        floored[z] += 1

    return floored


def compute_reposition_moves(current_alloc_list: List[str], desired_alloc: Dict[str, int]) -> List[Tuple[int, str]]:
    """Compute minimal moves to reach desired allocation.

    current_alloc_list: list of zones per elevator index, e.g. ['Lobby','Mid','Upper','Lobby']
    desired_alloc: dict zone->count

    Returns list of (elevator_index, target_zone) moves. Elevators already in a target zone
    may be left untouched. Greedy algorithm: find zones with surplus and deficits and move
    arbitrary elevators from surplus zones to deficit zones until satisfied.
    """
    n = len(current_alloc_list)
    # current counts
    curr_counts: Dict[str, int] = {z: 0 for z in ZONES}
    for z in current_alloc_list:
        curr_counts[z] = curr_counts.get(z, 0) + 1

    surplus_zones: List[str] = []
    deficit_zones: List[str] = []
    for z in ZONES:
        diff = curr_counts.get(z, 0) - desired_alloc.get(z, 0)
        if diff > 0:
            surplus_zones.extend([z] * diff)
        elif diff < 0:
            deficit_zones.extend([z] * (-diff))

    moves: List[Tuple[int, str]] = []
    # indices of candidate elevators in each surplus zone
    indices_by_zone: Dict[str, List[int]] = {z: [] for z in ZONES}
    for i, z in enumerate(current_alloc_list):
        if z in surplus_zones or curr_counts[z] > desired_alloc.get(z, 0):
            indices_by_zone[z].append(i)
    # Now pair surplus indices to deficits
    for target_zone in deficit_zones:
        # pick any surplus zone which still has an available elevator
        src = None
        for z in ZONES:
            if indices_by_zone.get(z):
                src = z
                break
        if src is None:
            break
        idx = indices_by_zone[src].pop()
        moves.append((idx, target_zone))
        # update counts
        curr_counts[src] -= 1
        curr_counts[target_zone] = curr_counts.get(target_zone, 0) + 1

    return moves


def parking_policy(state: str, y_hat: float, n_elevators: int = 4, current_alloc: List[str] | None = None) -> Dict[str, Any]:
    """Main entry: given traffic state and predicted demand, return desired allocation and moves.

    - state: one of the classifier labels from Task 2
    - y_hat: predicted number of calls next interval (float)
    - n_elevators: fleet size (default 4)
    - current_alloc: optional list of current parked zones per elevator; if None, default even spread

    Returns dict:
      {
        'desired_alloc': {'Lobby': int, 'Mid': int, 'Upper': int},
        'moves': [(elevator_index, target_zone), ...],
        'metadata': { 'proportions': {...}, 'y_hat': float(y_hat), 'state': state }
      }
    """
    if current_alloc is None:
        # default even spread (round-robin)
        current_alloc = [ZONES[i % len(ZONES)] for i in range(n_elevators)]

    desired = compute_desired_allocation(state, y_hat, n_elevators)
    moves = compute_reposition_moves(current_alloc, desired)

    return {
        'desired_alloc': desired,
        'moves': moves,
        'metadata': {
            'proportions': _proportions_from_state(state, y_hat),
            'y_hat': float(y_hat),
            'state': state,
            'n_elevators': n_elevators,
        }
    }


if __name__ == '__main__':
    # tiny local smoke demo
    print('Task3 module loaded. Call parking_policy(state,y_hat,...) to get allocations and moves.')
