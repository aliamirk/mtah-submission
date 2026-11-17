import numpy as np


def decode_two_state(evidence, penalty):
    """Viterbi over states home (0) and away (1).

    evidence: +1 where the evidence says home, -1 where it says away, 0 for no evidence.
    A frame whose evidence disagrees with the state costs 1, a switch costs `penalty`.
    Frames with no evidence cost nothing, so the state carries through gaps.
    """
    n = len(evidence)
    if n == 0:
        return np.empty(0, dtype=np.int8)
    c_home = np.where(evidence < 0, -evidence, 0.0).astype(np.float64)
    c_away = np.where(evidence > 0, evidence, 0.0).astype(np.float64)
    back = np.zeros((n, 2), dtype=np.int8)
    ch, ca = c_home[0], c_away[0]
    for t in range(1, n):
        if ch <= ca + penalty:
            nh, bh = ch, 0
        else:
            nh, bh = ca + penalty, 1
        if ca <= ch + penalty:
            na, ba = ca, 1
        else:
            na, ba = ch + penalty, 0
        back[t, 0], back[t, 1] = bh, ba
        ch, ca = nh + c_home[t], na + c_away[t]
    state = 0 if ch <= ca else 1
    out = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        out[t] = state
        state = back[t, state]
    return out


def decode_segments(evidence, active, penalty):
    """Decode each maximal run where `active` is True. -1 elsewhere."""
    out = np.full(len(evidence), -1, dtype=np.int8)
    edges = np.diff(np.concatenate(([0], active.astype(np.int8), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        out[a:b] = decode_two_state(evidence[a:b], penalty)
    return out
