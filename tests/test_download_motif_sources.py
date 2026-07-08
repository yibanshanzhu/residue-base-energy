from __future__ import annotations

import numpy as np

from scripts.download_motif_sources import (
    parse_hocomoco_jaspar_collection,
    parse_jaspar_collection,
    parse_jaspar_matrix,
)


def test_parse_jaspar_matrix_transposes_base_rows_to_pwm_rows():
    text = """>MA0001.1\tTEST
A  [ 1 2 ]
C  [ 3 4 ]
G  [ 5 6 ]
T  [ 7 8 ]
"""

    pwm = parse_jaspar_matrix(text)

    assert pwm.shape == (2, 4)
    assert np.allclose(pwm[0], [1, 3, 5, 7])
    assert np.allclose(pwm[1], [2, 4, 6, 8])


def test_parse_hocomoco_collection_uses_four_rows_per_motif():
    text = """>AHR_HUMAN.H11MO.0.B
1 2
3 4
5 6
7 8
>FOO_HUMAN.H11MO.0.A
10
20
30
40
"""

    motifs = parse_hocomoco_jaspar_collection(text)

    assert np.allclose(motifs["AHR_HUMAN.H11MO.0.B"], [[1, 3, 5, 7], [2, 4, 6, 8]])
    assert np.allclose(motifs["FOO_HUMAN.H11MO.0.A"], [[10, 20, 30, 40]])


def test_parse_jaspar_collection_indexes_multiple_records():
    text = """>UN0118.1\tELK4
A  [ 1 ]
C  [ 2 ]
G  [ 3 ]
T  [ 4 ]
>UN0124.1\tHOXB1
A  [ 5 ]
C  [ 6 ]
G  [ 7 ]
T  [ 8 ]
"""

    motifs = parse_jaspar_collection(text)

    assert np.allclose(motifs["UN0118.1"], [[1, 2, 3, 4]])
    assert np.allclose(motifs["UN0124.1"], [[5, 6, 7, 8]])
