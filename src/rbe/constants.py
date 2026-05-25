AA3_TO_1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}

AA_ORDER = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ORDER)}

DNA_RESNAMES = {"DA", "DC", "DG", "DT", "A", "C", "G", "T"}

DNA_BACKBONE_ATOMS = {
    "P",
    "OP1",
    "OP2",
    "OP3",
    "O1P",
    "O2P",
    "O3P",
    "C1'",
    "C2'",
    "C3'",
    "C4'",
    "C5'",
    "O2'",
    "O3'",
    "O4'",
    "O5'",
    "C1*",
    "C2*",
    "C3*",
    "C4*",
    "C5*",
    "O2*",
    "O3*",
    "O4*",
    "O5*",
}

BASE_ORDER = "ACGT"

