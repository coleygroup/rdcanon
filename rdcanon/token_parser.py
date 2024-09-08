import hashlib
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
import numpy as np
from lark import Lark, Transformer
from rdcanon.askcos_prims import prims as prims1
from rdcanon.pubchem_prims import prims as prims2
from rdcanon.drugbank_prims_with_nots import prims as prims3
from rdcanon.np_prims import prims as prims4
from functools import cmp_to_key
from rdcanon.rec_util import RecGraph
import re


# PRIMITIVE:  "D" | "H" | "h" | "R" | "r" | "v" | "X" | "x" | "-" | "+" | "#"
#                | "*" | "a" | "A" | "@" | "@@"
#                | "O" | "C" | "N" | "o" | "c" | "n" | "S" | "s" | "P" | "B" | "b" | "F" | "I" | "Cl" | "Br"
#                | "Se" | "Si" | "Sn" | "As" | "Te" | "Pb" | "Zn" | "Cu" | "Fe" | "Mg" | "Na" | "Ca" | "Al"
#                | "K" | "Li" | "Mn" | "Zr" | "Co" | "Ni" | "Cd" | "Ag" | "Au" | "Pt" | "Pd" | "Ru" | "Rh"
#                | "Ir" | "Ti" | "V" | "W" | "Mo" | "Hg" | "Tl" | "Bi" | "Ba" | "Sr" | "Cs" | "Rb" | "Be"


bond_value_map = {
    "UNSPECIFIED": 1000,
    "SINGLE": 901,
    "DOUBLE": 802,
    "TRIPLE": 700,
    "QUADRUPLE": 20,
    "QUINTUPLE": 19,
    "HEXTUPLE": 18,
    "ONEANDAHALF": 17,
    "TWOANDAHALF": 16,
    "THREEANDAHALF": 15,
    "FOURANDAHALF": 14,
    "FIVEANDAHALF": 13,
    "AROMATIC": 850,
    "IONIC": 12,
    "HYDROGEN": 11,
    "THREECENTER": 10,
    "DATIVEONE": 9,
    "DATIVE": 8,
    "DATIVEL": 7,
    "DATIVER": 6,
    "OTHER": 5,
    "ZERO": 4,
}


def hash_smarts(in_smarts, in_prims, func="sha256"):
    if func == "sha256":
        smarts_bytes = in_smarts.encode()
        hasher = hashlib.sha256()
        hasher.update(smarts_bytes)
        val = int(hasher.hexdigest(), 16)
    else:
        if in_smarts[0] == "!":
            if in_smarts in in_prims:
                val = in_prims[in_smarts]
            elif in_smarts[1:] not in in_prims:
                val = hash_smarts(in_smarts[1:], {}) / 1e78
            else:
                val = 1 / in_prims[in_smarts[1:]]
        else:
            if in_smarts not in in_prims:
                val = hash_smarts(in_smarts, {}) / 1e78
            else:
                val = in_prims[in_smarts]
    return val


prims1["*"] = 10e64
prims2["*"] = 10e64
prims3["*"] = 10e64
prims4["*"] = 10e64

for k in prims1:
    prims1[k] = prims1[k] + hash_smarts(k, {}) / 1e78
for k in prims2:
    prims2[k] = prims2[k] + hash_smarts(k, {}) / 1e78
for k in prims3:
    prims3[k] = prims3[k] + hash_smarts(k, {}) / 1e78
for k in prims4:
    prims4[k] = prims4[k] + hash_smarts(k, {}) / 1e78

labels = [
    "!",
    "*",
    "a",
    "A",
    "@",
    "@@",
    "C",
    "N",
    "O",
    "o",
    "c",
    "n",
    "s",
    "S",
    "P",
    "p",
    "B",
    "b",
    "F",
    "I",
    "Cl",
    "Br",
    "Se",
    "Si",
    "Sn",
    "As",
    "Te",
    "Pb",
    "Zn",
    "Cu",
    "Fe",
    "Mg",
    "Na",
    "Ca",
    "Al",
    "K",
    "Li",
    "Mn",
    "Zr",
    "Co",
    "Ni",
    "Cd",
    "Ag",
    "Au",
    "Pt",
    "Pd",
    "Ru",
    "Rh",
    "Ir",
    "Ti",
    "V",
    "W",
    "Mo",
    "Hg",
    "Tl",
    "Bi",
    "Ba",
    "Sr",
    "Cs",
    "Rb",
    "Be",
    "se",
    "te",
    "La",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Co",
    "Os",
    "Re",
    "Ga",
    "Ge",
    "Y",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Th",
    "Pa",
    "Mo",
    "U",
    "Tc",
    "At",
    "Am",
    "Bk",
    "Cf",
    "Cm",
    "He",
    "Ne",
    "Pm",
    "Pu",
    "Xe"
]

ATOMS = [
    "a",
    "A",
    "C",
    "N",
    "O",
    "o",
    "c",
    "n",
    "s",
    "S",
    "P",
    "p",
    "B",
    "b",
    "F",
    "I",
    "Cl",
    "Br",
    "Se",
    "Si",
    "Sn",
    "As",
    "Te",
    "Pb",
    "Zn",
    "Cu",
    "Fe",
    "Mg",
    "Na",
    "Ca",
    "Al",
    "K",
    "Li",
    "Mn",
    "Zr",
    "Co",
    "Ni",
    "Cd",
    "Ag",
    "Au",
    "Pt",
    "Pd",
    "Ru",
    "Rh",
    "Ir",
    "Ti",
    "V",
    "W",
    "Mo",
    "Hg",
    "Tl",
    "Bi",
    "Ba",
    "Sr",
    "Cs",
    "Rb",
    "Be",
    "se",
    "te",
    "La",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Co",
    "Os",
    "Re",
    "Ga",
    "Ge",
    "Y",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Th",
    "Pa",
    "Mo",
    "U",
    "Tc",
    "At",
    "Am",
    "Bk",
    "Cf",
    "Cm",
    "He",
    "Ne",
    "Pm",
    "Pu",
    "Xe"
]


class SMARTSTransformer2(Transformer):
    def start(self, args):
        results = []

        stack = deque()

        stack.append(args[0])

        seq = []
        while len(stack) > 0:
            cur = stack.popleft()
            if cur[0] == "deg":
                seq.append(cur)
                continue
            if cur[0] == "!":
                seq.append(cur[0])
                continue
            if cur in [
                "rec_start",
                "brack_start",
                "rec_end",
                "brack_end",
                "&",
                ",",
                ";",
            ]:
                seq.append(cur)
                continue
            if cur[0] == "open" or cur[0] == "close":
                seq.append(cur[0])
            if cur[0] == "component":
                seq.append(cur[1])
                continue
            if cur[0] == "bond":
                seq.append(cur[1])
                continue
            if cur[0] == "iso":
                seq.append(cur)
                continue
            if cur[0] == "nested":
                stack.appendleft("rec_start")
            if cur[0] == "bracketed":
                stack.appendleft("brack_start")
            for r in cur[1]:
                stack.appendleft(r)
            if cur[0] == "bracketed":
                stack.appendleft("brack_end")
            if cur[0] == "nested":
                stack.appendleft("rec_end")

        tok = ""
        for r in reversed(seq):
            if r[0] == "iso":
                tok = tok + r[1][0] + "*"
                continue
            if len(r) == 2:
                if r[0] == "deg":
                    tok = tok + str(r[1])
                    continue
            if r == "brack_start":
                tok = tok + "["
                continue
            if r == "brack_end":
                tok = tok + "]"
                continue
            if r == "rec_start":
                tok = tok + "$("
                continue
            if r == "rec_end":
                tok = tok + ")"
                continue
            if r == "open":
                tok = tok + "("
                continue
            if r == "close":
                tok = tok + ")"
                continue
            if r in ["=", "-", "#", "~", ":", ".", "@", "/", "\\", "!", "&", ",", ";"]:
                tok = tok + r
                continue
            else:
                for k in r:
                    if k[0] == "!":
                        tok = tok + k[0]
                    if k[0] == "atom":
                        for r2 in k[1]:
                            tok = tok + r2[1]
                    else:
                        for r2 in k[1]:
                            tok = tok + r2

        # if tok[0] == "[" and tok[-1] == "]":
        # tok = tok[1:-1]

        tok = tok[2:-1]

        # print("kek", tok)
        if tok[0] == "$":
            return tok, []

        # print(tok)

        atoms_in_seq = []
        bonds_in_seq = []
        cur_atom = ""
        cur_bond = ""
        reading = False
        atom_map = False
        rec = False
        nn = 0
        # print(seq)
        for r in reversed(seq):
            if nn == 0:
                nn = nn + 1
                continue
            # print("a", r, cur_atom, rec)
            if atom_map and r != "brack_end":
                cur_atom = cur_atom + r[1]
                atom_map = False
                continue
            if r == "rec_start":
                cur_atom = cur_atom + "$("
                # reading = True
                rec = True
                continue
            if r == "brack_start":
                if not rec:
                    if cur_bond != "":
                        bonds_in_seq.append(cur_bond)
                    cur_bond = ""
                    reading = True
                    continue
                else:
                    cur_atom = cur_atom + "["
                    continue

            if r == "brack_end":
                if not rec:
                    if atom_map:
                        atom_map = False
                    reading = False
                    if cur_atom != "":
                        atoms_in_seq.append(cur_atom)
                    cur_atom = ""
                    continue
                else:
                    cur_atom = cur_atom + "]"
                    continue

            if r == "rec_end":
                cur_atom = cur_atom + ")"
                # reading = False
                rec = False
                continue

            # print(reading)
            bond_symbs = ["-", "=", "#", "~", ":", ".", "@", "/", "\\"]
            op_symbs = ["!", "&", ",", ";"]

            if reading == False:
                # print("b", r)
                if r in op_symbs or r in bond_symbs:
                    cur_bond = cur_bond + r
                    continue
                if type(r[0]) == tuple and len(r) == 1:
                    if r[0][1][0][1][0] in bond_symbs:
                        cur_bond = cur_bond + r[0][1][0][1][0]
                        continue
                    cur_atom = cur_atom + r[0][1][0][1][0]
                    if cur_bond != "":
                        bonds_in_seq.append(cur_bond)
                        cur_bond = ""
                    atoms_in_seq.append(cur_atom)
                    cur_atom = ""
                    continue
                elif type(r[0]) == tuple and len(r) == 2:
                    cur_bond = cur_bond + r[0][1][0]
                    cur_atom = cur_atom + r[1][1][0][1][0]

                    bonds_in_seq.append(cur_bond)
                    cur_bond = ""

                    atoms_in_seq.append(cur_atom)
                    cur_atom = ""
                    continue

                else:
                    if r == "open":
                        continue
                    if r == "close":
                        continue
                    if r[0] == "deg" or r[0] == "bond":
                        cur_bond = cur_bond + r[1][0]
                        continue
            # # if reading:
            # print(r)
            # cur_atom = cur_atom + r[0]

            if (
                r == ";"
                or r == ":"
                or r == ","
                or r == "&"
                or r == "!"
                or r == "open"
                or r == "close"
                or r in bond_symbs
            ):
                if r == ":" and not rec:
                    atom_map = True
                if rec and r == "open":
                    cur_atom = cur_atom + "("
                    continue
                if rec and r == "close":
                    cur_atom = cur_atom + ")"
                    continue
                if rec and r in bond_symbs:
                    cur_atom = cur_atom + r
                    continue
                if not rec:
                    cur_atom = cur_atom + r
                    continue

            if r[0] == "deg":
                cur_atom = cur_atom + r[1]
                continue
            # print("c",r)

            if rec and r in op_symbs:
                cur_atom = cur_atom + r
                continue

            result = {
                "!": -1,
                "D": -1,
                "H": -1,
                "h": -1,
                "R": -1,
                "r": -1,
                "v": -1,
                "X": -1,
                "x": -1,
                "-": -1,
                "+": -1,
                "#": -1,
                "*": -1,
                "a": -1,
                "A": -1,
                "@": -1,
                "@@": -1,
                "bond": -1,
                "rec": -1,
                "C": -1,
                "N": -1,
                "O": -1,
                "o": -1,
                "c": -1,
                "n": -1,
                "S": -1,
                "s": -1,
                "P": -1,
                "p": -1,
                "B": -1,
                "b": -1,
                "F": -1,
                "I": -1,
                "Cl": -1,
                "Br": -1,
                "Se": -1,
                "Si": -1,
                "Sn": -1,
                "As": -1,
                "Te": -1,
                "Pb": -1,
                "Zn": -1,
                "Cu": -1,
                "Fe": -1,
                "Mg": -1,
                "Na": -1,
                "Ca": -1,
                "Al": -1,
                "K": -1,
                "Li": -1,
                "Mn": -1,
                "Zr": -1,
                "Co": -1,
                "Ni": -1,
                "Cd": -1,
                "Ag": -1,
                "Au": -1,
                "Pt": -1,
                "Pd": -1,
                "Ru": -1,
                "Rh": -1,
                "Ir": -1,
                "Ti": -1,
                "V": -1,
                "W": -1,
                "Mo": -1,
                "Hg": -1,
                "Tl": -1,
                "Bi": -1,
                "Ba": -1,
                "Sr": -1,
                "Cs": -1,
                "Rb": -1,
                "Be": -1,
                "se": -1,
                "te": -1,
                "La": -1,
                "Er": -1,
                "Tm": -1,
                "Yb": -1,
                "Lu": -1,
                "Hf": -1,
                "Ta": -1,
                "W" : -1,
                "Re": -1,
                "Co": -1,
                "Os": -1,
                "Re": -1,
                "Ga": -1,
                "Ge": -1,
                "Y": -1,
                "Ce": -1,
                "Pr": -1,
                "Nd": -1,
                "Sm": -1,
                "Eu": -1,
                "Gd": -1,
                "Tb": -1,
                "Dy": -1,
                "Ho": -1,
                "Th": -1,
                "Pa": -1,
                "Mo": -1,
                "U": -1,
                "Tc": -1,
                "At": -1,
                "Am": -1,
                "Bk": -1,
                "Cf": -1,
                "Cm": -1,
                "He": -1,
                "Ne": -1,
                "Pm": -1,
                "Pu": -1,
                "Xe": -1,
                "iso": -1,
            }

            if r[0] == "iso":
                result["iso"] = r[1][0]
                results.append(result)
                cur_atom = cur_atom + r[1][0] + "*"
                continue

            if len(r) == 2:
                result["bond"] = str(r[0][1])
                atms = r[1][1]
            else:
                atms = r[0][1]

            if rec:
                if atms[0][1][0] in bond_symbs:
                    if len(atms) == 1:
                        cur_atom = cur_atom + atms[0][1][0]
                    else:
                        cur_atom = cur_atom + atms[0][1][0]
                        cur_atom = cur_atom + atms[1][1][0]
                    continue

            deg_found = False
            prev_prim = False
            for rr in atms:

                if rr[0] == "!":
                    result["!"] = True
                    continue
                if prev_prim and rr[0] != "deg":
                    if k == "R" or k == "h" or k == "r" or k == "x":
                        result[k] = "default"
                    else:
                        result[k] = 1
                    deg_found = True
                if rr[0] == "prim":
                    k = rr[1]
                    prev_prim = True
                    deg_found = False
                elif rr[0] == "deg":
                    deg_found = True
                    if result[k] == -1:
                        result[k] = int(rr[1])
                    else:
                        result[k] = result[k] + int(rr[1])
                    prev_prim = False
            if not deg_found:
                if result[k] == -1:
                    if k == "R" or k == "h" or k == "r" or k == "x":
                        result[k] = "default"
                    else:
                        result[k] = 1
                else:
                    result[k] = result[k] + 1
            # print(k, result[k])

            cur_atom = cur_atom + parse_label(result)
            # print(cur_atom)

            results.append(result)
        return atoms_in_seq, bonds_in_seq

    def isotope(self, args):
        return "iso", args

    def not1(self, args):
        return "!", args[0]  # Returning '!' to indicate its presence

    def not2(self, args):
        return "!", args  # Returning '!' to indicate its presence

    def symbol(self, args):
        return "prim", args[0]

    def symbol_single(self, args):
        return "prim", args[0]

    def degree1(self, args):
        return "deg", args[0]

    def nested_rule(self, args):
        return "nested", args

    def item(self, args):
        return "item", args

    def component(self, args):
        return "component", args

    def atom(self, args):
        return "atom", args

    def token(self, args):
        return "token", args

    def operator_symbol(self, args):
        return "op", args[0]

    def bond_symbol(self, args):
        return "bond", args[0]

    def open(self, args):
        return "open", args

    def close(self, args):
        return "close", args

    def bracketed_rule(self, args):
        return "bracketed", args

    def BOND_PRIMITIVE(self, args):
        return args

    def PRIMITIVE(self, args):
        return args

    def DIGIT(self, args):
        return args

    def NOT(self, args):
        return args

    def OPERATOR_PRIMITIVE(self, args):
        return args


class SMARTSTransformer(Transformer):
    def start(self, args):
        results = []

        stack = deque()

        stack.append(args[0])

        seq = []
        while len(stack) > 0:
            cur = stack.popleft()
            if cur[0] == "deg":
                seq.append(cur)
                continue
            if cur[0] == "!":
                seq.append(cur[0])
                continue
            if cur in [
                "rec_start",
                "brack_start",
                "rec_end",
                "brack_end",
                "&",
                ",",
                ";",
            ]:
                seq.append(cur)
                continue
            if cur[0] == "open" or cur[0] == "close":
                seq.append(cur[0])
            if cur[0] == "component":
                seq.append(cur[1])
                continue
            if cur[0] == "bond":
                seq.append(cur[1])
                continue
            if cur[0] == "iso":
                seq.append(cur)
                continue
            if cur[0] == "nested":
                stack.appendleft("rec_start")
            if cur[0] == "bracketed":
                stack.appendleft("brack_start")
            for r in cur[1]:
                stack.appendleft(r)
            if cur[0] == "bracketed":
                stack.appendleft("brack_end")
            if cur[0] == "nested":
                stack.appendleft("rec_end")

        tok = ""
        for r in reversed(seq):
            if r[0] == "iso":
                tok = tok + r[1][0] + "*"
                continue
            if len(r) == 2:
                if r[0] == "deg":
                    tok = tok + str(r[1])
                    continue
            if r == "brack_start":
                tok = tok + "["
                continue
            if r == "brack_end":
                tok = tok + "]"
                continue
            if r == "rec_start":
                tok = tok + "$("
                continue
            if r == "rec_end":
                tok = tok + ")"
                continue
            if r == "open":
                tok = tok + "("
                continue
            if r == "close":
                tok = tok + ")"
                continue
            if r in ["=", "-", "#", "~", ":", ".", "@", "/", "!", "&", ",", ";"]:
                tok = tok + r
                continue
            else:
                for k in r:
                    if k[0] == "!":
                        tok = tok + k[0]
                    if k[0] == "atom":
                        for r2 in k[1]:
                            tok = tok + r2[1]
                    else:
                        for r2 in k[1]:
                            tok = tok + r2

        if tok[0] == "[" and tok[-1] == "]":
            tok = tok[1:-1]

        if "$" not in tok:
            for r in reversed(seq):
                result = {
                    "!": -1,
                    "D": -1,
                    "H": -1,
                    "h": -1,
                    "R": -1,
                    "r": -1,
                    "v": -1,
                    "X": -1,
                    "x": -1,
                    "-": -1,
                    "+": -1,
                    "#": -1,
                    "*": -1,
                    "a": -1,
                    "A": -1,
                    "@": -1,
                    "@@": -1,
                    "bond": -1,
                    "rec": -1,
                    "C": -1,
                    "N": -1,
                    "O": -1,
                    "o": -1,
                    "c": -1,
                    "n": -1,
                    "S": -1,
                    "s": -1,
                    "P": -1,
                    "p": -1,
                    "B": -1,
                    "b": -1,
                    "F": -1,
                    "I": -1,
                    "Cl": -1,
                    "Br": -1,
                    "Se": -1,
                    "Si": -1,
                    "Sn": -1,
                    "As": -1,
                    "Te": -1,
                    "Pb": -1,
                    "Zn": -1,
                    "Cu": -1,
                    "Fe": -1,
                    "Mg": -1,
                    "Na": -1,
                    "Ca": -1,
                    "Al": -1,
                    "K": -1,
                    "Li": -1,
                    "Mn": -1,
                    "Zr": -1,
                    "Co": -1,
                    "Ni": -1,
                    "Cd": -1,
                    "Ag": -1,
                    "Au": -1,
                    "Pt": -1,
                    "Pd": -1,
                    "Ru": -1,
                    "Rh": -1,
                    "Ir": -1,
                    "Ti": -1,
                    "V": -1,
                    "W": -1,
                    "Mo": -1,
                    "Hg": -1,
                    "Tl": -1,
                    "Bi": -1,
                    "Ba": -1,
                    "Sr": -1,
                    "Cs": -1,
                    "Rb": -1,
                    "Be": -1,
                    "se": -1,
                    "te": -1,
                    "La": -1,
                    "Er": -1,
                    "Tm": -1,
                    "Yb": -1,
                    "Lu": -1,
                    "Hf": -1,
                    "Ta": -1,
                    "W": -1,
                    "Re": -1,
                    "Co": -1,
                    "Os": -1,
                    "Re": -1,
                    "Ga": -1,
                    "Ge": -1,
                    "Y": -1,
                    "Ce": -1,
                    "Pr": -1,
                    "Nd": -1,
                    "Sm": -1,
                    "Eu": -1,
                    "Gd": -1,
                    "Tb": -1,
                    "Dy": -1,
                    "Ho": -1,
                    "Th": -1,
                    "Pa": -1,
                    "Mo": -1,
                    "U": -1,
                    "Tc": -1,
                    "At": -1,
                    "Am": -1,
                    "Bk": -1,
                    "Cf": -1,
                    "Cm": -1,
                    "He": -1,
                    "Ne": -1,
                    "Pm": -1,
                    "Pu": -1,
                    "Xe": -1,
                    "iso": -1,
                }

                if r[0] == "iso":
                    result["iso"] = r[1][0]
                    results.append(result)
                    continue

                if len(r) == 2:
                    result["bond"] = str(r[0][1])
                    atms = r[1][1]
                else:
                    atms = r[0][1]

                deg_found = False
                prev_prim = False
                for rr in atms:
                    if rr[0] == "!":
                        result["!"] = True
                        continue
                    if prev_prim and rr[0] != "deg":
                        if k == "R" or k == "h" or k == "r" or k == "x":
                            result[k] = "default"
                        else:
                            result[k] = 1
                        deg_found = True
                    if rr[0] == "prim":
                        k = rr[1]
                        prev_prim = True
                        deg_found = False
                    elif rr[0] == "deg":
                        deg_found = True
                        if result[k] == -1:
                            result[k] = int(rr[1])
                        else:
                            result[k] = result[k] + int(rr[1])
                        prev_prim = False
                if not deg_found:
                    if result[k] == -1:
                        if k == "R" or k == "h" or k == "r" or k == "x":
                            result[k] = "default"
                        else:
                            result[k] = 1
                    else:
                        result[k] = result[k] + 1
                results.append(result)
        else:
            result = {
                "!": -1,
                "D": -1,
                "H": -1,
                "h": -1,
                "R": -1,
                "r": -1,
                "v": -1,
                "X": -1,
                "x": -1,
                "-": -1,
                "+": -1,
                "#": -1,
                "*": -1,
                "a": -1,
                "A": -1,
                "@": -1,
                "@@": -1,
                "bond": -1,
                "rec": 1,
                "C": -1,
                "N": -1,
                "O": -1,
                "o": -1,
                "c": -1,
                "n": -1,
                "S": -1,
                "s": -1,
                "P": -1,
                "p": -1,
                "B": -1,
                "b": -1,
                "F": -1,
                "I": -1,
                "Cl": -1,
                "Br": -1,
                "Se": -1,
                "Si": -1,
                "Sn": -1,
                "As": -1,
                "Te": -1,
                "Pb": -1,
                "Zn": -1,
                "Cu": -1,
                "Fe": -1,
                "Mg": -1,
                "Na": -1,
                "Ca": -1,
                "Al": -1,
                "K": -1,
                "Li": -1,
                "Mn": -1,
                "Zr": -1,
                "Co": -1,
                "Ni": -1,
                "Cd": -1,
                "Ag": -1,
                "Au": -1,
                "Pt": -1,
                "Pd": -1,
                "Ru": -1,
                "Rh": -1,
                "Ir": -1,
                "Ti": -1,
                "V": -1,
                "W": -1,
                "Mo": -1,
                "Hg": -1,
                "Tl": -1,
                "Bi": -1,
                "Ba": -1,
                "Sr": -1,
                "Cs": -1,
                "Rb": -1,
                "Be": -1,
                "se": -1,
                "te": -1,
                "La": -1,
                "Er": -1,
                "Tm": -1,
                "Yb": -1,
                "Lu": -1,
                "Hf": -1,
                "Ta": -1,
                "W": -1,
                "Re": -1,
                "Co": -1,
                "Os": -1,
                "Re": -1,
                "Ga": -1,
                "Ge": -1,
                "Y": -1,
                "Ce": -1,
                "Pr": -1,
                "Nd": -1,
                "Sm": -1,
                "Eu": -1,
                "Gd": -1,
                "Tb": -1,
                "Dy": -1,
                "Ho": -1,
                "Th": -1,
                "Pa": -1,
                "Mo": -1,
                "U": -1,
                "Tc": -1,
                "At": -1,
                "Am": -1,
                "Bk": -1,
                "Cf": -1,
                "Cm": -1,
                "He": -1,
                "Ne": -1,
                "Pm": -1,
                "Pu": -1,
                "Xe": -1,
                "iso": -1,
            }
            results.append(result)

        return results, tok

    def isotope(self, args):
        return "iso", args

    def not1(self, args):
        return "!", args[0]  # Returning '!' to indicate its presence

    def not2(self, args):
        return "!", args  # Returning '!' to indicate its presence

    def symbol(self, args):
        return "prim", args[0]

    def symbol_single(self, args):
        return "prim", args[0]

    def degree1(self, args):
        return "deg", args[0]

    def nested_rule(self, args):
        return "nested", args

    def item(self, args):
        return "item", args

    def component(self, args):
        return "component", args

    def atom(self, args):
        return "atom", args

    def token(self, args):
        return "token", args

    def operator_symbol(self, args):
        return "op", args[0]

    def bond_symbol(self, args):
        return "bond", args[0]

    def open(self, args):
        return "open", args

    def close(self, args):
        return "close", args

    def bracketed_rule(self, args):
        return "bracketed", args

    def BOND_PRIMITIVE(self, args):
        return args

    def PRIMITIVE(self, args):
        return args

    def DIGIT(self, args):
        return args

    def NOT(self, args):
        return args

    def OPERATOR_PRIMITIVE(self, args):
        return args


def split_smarts_f(out_sm):
    if out_sm[0] == "[" and out_sm[-1] == "]":
        deb = out_sm[1:-1]
    else:
        deb = out_sm
    debsp = custom_split(deb, ";")
    return debsp


def custom_split(input_string, delimiter=";", nested_start="$(", nested_end=")"):
    """
    Splits the input_string by the specified delimiter, but does not split anything within the nested structures.
    """
    parts = []
    current = []
    nested_level = 0
    i = 0
    while i < len(input_string):
        if input_string.startswith(nested_start, i):
            nested_level += 1
            current.append(input_string[i])
        elif input_string.startswith("(", i) and not input_string.startswith(
            nested_start, i - 1
        ):
            nested_level += 1
            current.append(input_string[i])
        elif input_string.startswith(nested_end, i) and nested_level:
            nested_level -= 1
            current.append(input_string[i])
        elif input_string[i] == delimiter and nested_level == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(input_string[i])
        i += 1
    if current:
        parts.append("".join(current))
    return parts


def categorize_string(sm, groups, nested_start="$(", nested_end=")"):
    """
    Categorize the string based on the presence of specific characters outside of nested structures.
    """
    nested_level = 0
    found_chars = set()
    i = 0
    while i < len(sm):
        if sm.startswith(nested_start, i) or sm.startswith("(", i):
            nested_level += 1
            i += len(nested_start) - 1
        elif sm.startswith(nested_end, i) and nested_level > 0:
            nested_level -= 1
            i += len(nested_end) - 1

        if nested_level == 0 and sm[i] in {"!", "&", ","}:
            found_chars.add(sm[i])

        i += 1
    # Categorize based on found characters
    key = "".join(sorted(found_chars)) if found_chars else "p"
    groups[key].append(sm)


def group_split_smarts(split_smarts):
    # start with queries containing ! and & and ,
    # then add queries containing only ! and &
    # then add queries containing only ! and ,
    # then add queries containing only !
    # then add queries containing only & and ,
    # then add queries containing only &
    # then add queries containing only ,

    groups = {
        "!&,": [],
        "!&": [],
        "!,": [],
        "!": [],
        "&,": [],
        "&": [],
        ",": [],
        "p": [],
    }
    for sm in split_smarts:
        categorize_string(sm, groups)
    return groups


def check_special_chars_outside_nested(sm, nested_start="$(", nested_end=")"):
    """
    Check if "&" is in the string `sm`, but not inside `$( x )` structures.
    """
    inside_nested = False
    found_chars = {"&": False}
    i = 0

    nested_level = 0
    while i < len(sm):
        if sm.startswith(nested_start, i) or sm.startswith("(", i):
            nested_level += 1
            i += len(nested_start) - 1
        elif sm.startswith(nested_end, i) and nested_level > 0:
            nested_level -= 1
            i += len(nested_end) - 1

        if nested_level == 0 and sm[i] in found_chars:
            found_chars[sm[i]] = True

        i += 1
    return all(found_chars.values())


def reorder_and_token(in_token):
    tokens = []
    spl_token = custom_split(in_token, "&")
    for token in spl_token:
        # primitive
        if token[0] != "[" and token[-1] != "]":
            token = "[" + token + "]"
        prim_tree = parser.parse(token)
        tokens.append((prim_tree,))
    return ("and", tokens)


def reorder_comma_token(in_token):
    tokens = []
    spl_token = custom_split(in_token, ",")
    for token in spl_token:
        if check_special_chars_outside_nested(token):
            rtoken = reorder_and_token(token)
            tokens.append(rtoken)
        else:
            # primitive
            if token[0] != "[" and token[-1] != "]":
                token = "[" + token + "]"
            prim_tree = parser.parse(token)
            tokens.append((prim_tree,))
    return ("or", tokens)


def reorder_internal_split_smarts(grouped_split_smarts):
    ordered_out = {}
    for group in all_groups:
        ordered_out[group] = []
    for group in all_groups:
        if group == "!&,":
            for smt in grouped_split_smarts[group]:
                tok_out = reorder_comma_token(smt)
                ordered_out[group].append(tok_out)
        elif group == "!&":
            for smt in grouped_split_smarts[group]:
                tok_out = reorder_and_token(smt)
                ordered_out[group].append(tok_out)
        elif group == "!,":
            for smt in grouped_split_smarts[group]:
                tok_out = reorder_comma_token(smt)
                ordered_out[group].append(tok_out)
        elif group == "!":
            # primitive
            for smt in grouped_split_smarts[group]:
                if smt[0] != "[" and smt[-1] != "]":
                    smt = "[" + smt + "]"
                prim_tree = parser.parse(smt)
                ordered_out[group].append((prim_tree,))
        elif group == "&,":
            for smt in grouped_split_smarts[group]:
                tok_out = reorder_comma_token(smt)
                ordered_out[group].append(tok_out)
        elif group == "&":
            for smt in grouped_split_smarts[group]:
                tok_out = reorder_and_token(smt)
                ordered_out[group].append(tok_out)
        elif group == ",":
            for smt in grouped_split_smarts[group]:
                tok_out = reorder_comma_token(smt)
                ordered_out[group].append(tok_out)
        else:
            # primitive
            for smt in grouped_split_smarts[group]:
                if smt[0] != "[" and smt[-1] != "]":
                    smt = "[" + smt + "]"
                prim_tree = parser.parse(smt)
                ordered_out[group].append((prim_tree,))
    return ordered_out


def sanitize_smarts_token(in_token):
    o1 = list(set(split_smarts_f(in_token)))
    o2 = group_split_smarts(o1)
    o3 = reorder_internal_split_smarts(o2)
    return o3, o2


def parse_label(lab):
    tx = ""
    if lab["rec"] == 1:
        return "rec"
    elif lab["iso"] != -1:
        return lab["iso"] + "*"
    for r in lab:
        if lab[r] == -1:
            continue
        if r in labels:
            tx = tx + r
        else:
            if lab[r] == "default":
                tx = tx + r
            else:
                tx = tx + r + str(lab[r])
    return tx


def gen_data_substructure(tree_in, digraph, prims):
    results = []
    ops = []
    stack = deque()
    stack.append((tree_in, 0))
    n = len(digraph.nodes)
    while len(stack) > 0:
        cur = stack.popleft()
        weights = []
        if cur[0][0] == "and":
            label = "&"
            for iidx, i in enumerate(cur[0][1]):
                if iidx != len(cur[0][1]) - 1:
                    ops.append("and (&)")
                stack.append((i, n))
        elif cur[0][0] == "or":
            label = ","
            for iidx, i in enumerate(cur[0][1]):
                stack.append((i, n))
                if iidx != len(cur[0][1]) - 1:
                    ops.append("or (,)")
        else:
            result, tok = transformer.transform(cur[0][0])
            label = parse_label(result[0])
            if label == "rec":
                if tok[0] == "!":
                    inc = 1
                else:
                    inc = 0
                mm_rec = tok[2 + inc : -1]
                if inc:
                    rg = RecGraph(recursive_compare)
                    rg.graph_from_smarts(mm_rec, order_token_canon, prims)
                    o, w = rg.recreate_molecule()
                    weights = w
                    label = "!$(" + o + ")"

                else:
                    rg = RecGraph(recursive_compare)
                    rg.graph_from_smarts(mm_rec, order_token_canon, prims)
                    o, w = rg.recreate_molecule()
                    weights = w
                    label = "$(" + o + ")"

            for res in result:
                results.append(res)

        if len(weights) > 0:
            digraph.add_node(n, label=label, weights=[tuple(weights)])
        else:
            digraph.add_node(n, label=label)

        digraph.add_edge(n, cur[1])
        n = n + 1

    x_toks = results[0].keys()
    hm = []
    ops.append("")
    for rr in results:
        if type(rr) == str:
            continue
        ar = []
        for x in x_toks:
            if rr[x] == -1:
                ar.append(np.nan)
            else:
                ar.append(rr[x])
        hm.append(ar)
    hm = np.array(hm).T

    return hm, ops, x_toks


def gen_data_structure(sanitized, group_smarts, test_smarts, prims="askcos"):
    trees = []
    titles = []

    token_num = 1
    for ixiii, r in enumerate(sanitized):  # group
        if len(sanitized[r]) == 0:
            continue
        for ixi, i in enumerate(sanitized[r]):  # sample
            token_num += 1
            trees.append(i)
            titles.append(group_smarts[r][ixi])

    digraph = nx.DiGraph()
    digraph.add_node(0, label=";")
    hmps = []
    opss = []
    x_tokss = []
    for ix, i in enumerate(trees):
        hm, ops, x_toks = gen_data_substructure(i, digraph, prims)
        hmps.append(hm)
        opss.append(ops)
        x_tokss.append(x_toks)

    return hmps, opss, x_tokss, digraph, titles


all_groups = ["!&,", "!&", "!,", "!", "&,", "&", ",", "p"]

grammar = """
start: "[" item* "]"

item: (open item* close | not2? nested_rule | component | bond_symbol nested_rule | bracketed_rule | bond_symbol | degree1 | isotope ) operator_symbol?

open: "("
close: ")"

component: (bond_symbol atom) | atom

atom: not1? symbol degree1? 

isotope: INTEGER "*"

bracketed_rule: ( "[" item* "]" )

nested_rule: "$(" item* ")"

not2: "!"

not1: NOT

NOT: "!"

symbol: PRIMITIVE

PRIMITIVE:  "D" | "H" | "h" | "R" | "r" | "v" | "X" | "x" | "-" | "+" | "#" 
                | "*" | "a" | "A" | "@" | "@@"
                | "O" | "C" | "N" | "o" | "c" | "n" | "S" | "s" | "P" | "p" | "B" | "b" | "F" | "I" | "Cl" | "Br"
                | "Se" | "Si" | "Sn" | "As" | "Te" | "Pb" | "Zn" | "Cu" | "Fe" | "Mg" | "Na" | "Ca" | "Al"
                | "K" | "Li" | "Mn" | "Zr" | "Co" | "Ni" | "Cd" | "Ag" | "Au" | "Pt" | "Pd" | "Ru" | "Rh"
                | "Ir" | "Ti" | "V" | "W" | "Mo" | "Hg" | "Tl" | "Bi" | "Ba" | "Sr" | "Cs" | "Rb" | "Be" | "se" | "te"
                | "La" | "Er" | "Tm" | "Yb" | "Lu" | "Hf" | "Ta" | "W" | "Re" | "Co" | "Os" | "Re" | "Ga" | "Ge" | "Y"
                | "Ce" | "Pr" | "Nd" | "Sm" | "Eu" | "Gd" | "Tb" | "Dy" | "Ho" | "Th" | "Pa" | "Mo" | "U" | "Tc" | "At"
                | "Am" | "Bk" | "Cf" | "Cm" | "He" | "Ne" | "Pm" | "Pu" | "Xe"

symbol_single: PRIMITIVE_SINGLE
                
PRIMITIVE_SINGLE:  "*" | "a" | "A" | "@" | "@@"
                    | "O" | "C" | "N" | "o" | "c" | "n" | "S" | "s" | "P" | "p" | "B" | "b" | "F" | "I" | "Cl" | "Br"
                    | "Se" | "Si" | "Sn" | "As" | "Te" | "Pb" | "Zn" | "Cu" | "Fe" | "Mg" | "Na" | "Ca" | "Al"
                    | "K" | "Li" | "Mn" | "Zr" | "Co" | "Ni" | "Cd" | "Ag" | "Au" | "Pt" | "Pd" | "Ru" | "Rh"
                    | "Ir" | "Ti" | "V" | "W" | "Mo" | "Hg" | "Tl" | "Bi" | "Ba" | "Sr" | "Cs" | "Rb" | "Be" | "se" | "te"
                    | "La" | "Er" | "Tm" | "Yb" | "Lu" | "Hf" | "Ta" | "W" | "Re" | "Co" | "Os" | "Re" | "Ga" | "Ge" | "Y"
                    | "Ce" | "Pr" | "Nd" | "Sm" | "Eu" | "Gd" | "Tb" | "Dy" | "Ho" | "Th" | "Pa" | "Mo" | "U" | "Tc" | "At"
                    | "Am" | "Bk" | "Cf" | "Cm" | "He" | "Ne" | "Pm" | "Pu" | "Xe"

PRIMITIVE_MULTI:  "D" | "H" | "h" | "R" | "r" | "v" | "X" | "x" | "-" | "+" | "#" 

symbol_multi: PRIMITIVE_MULTI

bond_symbol: BOND_PRIMITIVE

BOND_PRIMITIVE: "-" | "=" | "#" | "~" | ":" | "." | "@" | "/" | "\\\\"

operator_symbol: OPERATOR_PRIMITIVE

OPERATOR_PRIMITIVE: ";" | "," | "&"

degree1: INTEGER

INTEGER: DIGIT+

DIGIT: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

%ignore " "
"""

parser = Lark(grammar, parser="lalr")
transformer = SMARTSTransformer()

transformer2 = SMARTSTransformer2()


def parse_smarts_total(in_smarts, num_atoms):
    if in_smarts[0:2] == "[$" and num_atoms == 1:
        parsed = parser.parse(in_smarts)
    else:
        parsed = parser.parse("[$(" + in_smarts + ")]")

    atoms_seq, bonds_seq = transformer2.transform(parsed)
    return atoms_seq, bonds_seq


def recursive_compare(list1, list2):
    index1 = index2 = 0

    if not list2 or not list1:
        return True

    while index1 < len(list1) and index2 < len(list2):
        sub_item1, sub_item2 = list1[index1], list2[index2]
        if isinstance(sub_item1, list) and isinstance(sub_item2, list):
            result = recursive_compare(sub_item1, sub_item2)
            if result != 0:
                return result
            index1 += 1
            index2 += 1
        elif isinstance(sub_item1, tuple) and isinstance(sub_item2, tuple):
            result = recursive_compare(sub_item1, sub_item2)
            if result != 0:
                return result
            index1 += 1
            index2 += 1
        elif isinstance(sub_item1, list) and isinstance(sub_item2, tuple):
            result = recursive_compare(sub_item1, [sub_item2])
            if result != 0:
                return result
            index2 += 1
        elif isinstance(sub_item1, tuple) and isinstance(sub_item2, list):
            result = recursive_compare([sub_item1], sub_item2)
            if result != 0:
                return result
            index1 += 1
        elif isinstance(sub_item1, list):
            result = recursive_compare(sub_item1, [sub_item2])
            if result != 0:
                return result
            index2 += 1
        elif isinstance(sub_item1, tuple):
            result = recursive_compare(sub_item1, [sub_item2])
            if result != 0:
                return result
            index2 += 1
        elif isinstance(sub_item2, list) or isinstance(sub_item2, tuple):
            result = recursive_compare([sub_item1], sub_item2)
            if result != 0:
                return result
            index1 += 1
        else:
            if sub_item1 != sub_item2:
                return (sub_item1 > sub_item2) - (sub_item1 < sub_item2)
            index1 += 1
            index2 += 1

    return (len(list1) > len(list2)) - (len(list1) < len(list2))


def custom_key2(item1t, item2t):
    item1, tiebreaker1 = item1t
    item2, tiebreaker2 = item2t
    return recursive_compare(item1, item2)


def custom_key(item1t, item2t):
    item1, tiebreaker1 = item1t
    item2, tiebreaker2 = item2t
    return recursive_compare(item1, item2)


def moveToFront(lst, pos):
    if pos >= len(lst) or pos < 0:
        raise IndexError("Position is out of the range of the list")

    element = lst.pop(pos)
    lst.insert(0, element)

    return lst


def order_token_canon(
    in_smarts_token="[!a@H&D2;#7,#6;H;a-3;#7,!O,!#8&!O;#7,!O,!#8&!O++;*;H0]",
    atom_map=None,
    embedding="drugbank",
    min_num_explicit_hs=None,
    opt_num_explicit_hs=None,
):
    if embedding == "askcos":
        prims = prims1
    elif embedding == "pubchem":
        prims = prims2
    elif embedding == "drugbank":
        prims = prims3
    elif embedding == "npatlas":
        prims = prims4
    else:
        if type(embedding) == dict:
            prims = embedding
        else:
            raise ValueError(
                "embedding must be 'askcos', 'pubchem', 'drugbank', 'npatlas', or a dictionary of primitives"
            )

    # print(in_smarts_token)
    sanitized, group_smarts = sanitize_smarts_token(in_smarts_token)
    # print(sanitized, group_smarts)
    _, _, _, dg, _ = gen_data_structure(sanitized, group_smarts, in_smarts_token, prims)

    nodes_to_remove = []
    for nn_node in dg.nodes:
        if dg.nodes[nn_node]["label"] == "&":
            in_nodes = list(dg.in_edges(nn_node))
            seen = []
            for nn_in in in_nodes:
                # print(nn_node, nn_in[0], dg.nodes[nn_in[0]]["label"])
                if dg.nodes[nn_in[0]]["label"] in seen:
                    nodes_to_remove.append(nn_in[0])
                else:
                    seen.append(dg.nodes[nn_in[0]]["label"])
    #         # remove
    #         # nodes_to_remove.append(nn_node)
    # print()
    for node in nodes_to_remove:
        dg.remove_node(node)

    # add "H" + num_explicit_hs to the root node 0
    # print(dg.edges, [dg.nodes[i] for i in dg.nodes])
    # print()
    if min_num_explicit_hs != None and opt_num_explicit_hs==None:
        dg.add_node(len(dg.nodes), label="H" + str(min_num_explicit_hs))
        dg.add_edge(len(dg.nodes) - 1, 0)
    elif min_num_explicit_hs != None and opt_num_explicit_hs != None:
        dg.add_node(len(dg.nodes), label=",")
        idx_new = len(dg.nodes) - 1
        dg.add_edge(idx_new, 0)
        idx_new_1 = len(dg.nodes)
        idx_new_2 = len(dg.nodes) + 1

        dg.add_node(idx_new_1, label="H" + str(min_num_explicit_hs))
        dg.add_edge(idx_new_1, idx_new)
        dg.add_node(idx_new_2, label="H" + str(opt_num_explicit_hs+min_num_explicit_hs))
        dg.add_edge(idx_new_2, idx_new)
    # print(dg.edges, [dg.nodes[i] for i in dg.nodes])
    # print(dg.nodes, [dg.nodes[i] for i in dg.nodes])

    remaining_nodes = []
    for node in dg.nodes():
        if dg.in_degree(node) == 0:
            prim_sm = dg.nodes[node]["label"]
            if prim_sm[0] == "!":
                inc = 1
            else:
                inc = 0
            if prim_sm[0 + inc] == "$":
                pass
            else:
                dg.nodes[node]["weights"] = [
                    hash_smarts(prim_sm, prims, func="embedded")
                ]

            dg.nodes[node]["text"] = prim_sm
        else:
            remaining_nodes.append(node)

    stack = deque()
    stack.append(0)
    # print(dg.edges, [dg.nodes[i] for i in dg.nodes])
    # print(dg.in_edges(0))

    weights_in_order = []
    been_sorted = []
    while stack:
        node = stack.popleft()
        if node in been_sorted:
            continue
        these_weights = []
        recurse = False

        op = dg.nodes[node]["label"]
        # print("node", node)
        for neighbor in dg.in_edges(node):
            adj = neighbor[0]
            # print("neh", adj, dg.nodes[adj])
            if "weights" not in dg.nodes[adj]:
                stack.append(adj)
                stack.append(node)
                recurse = True
                break
            else:
                if "text" in dg.nodes[adj]:
                    txt = dg.nodes[adj]["text"]
                else:
                    txt = None

                these_weights.append((dg.nodes[adj]["weights"], txt))

        if not recurse:
            been_sorted.append(node)

            if op == "&":
                or_found = False
                for neighbor in dg.out_edges(node):
                    adj = neighbor[1]
                    if dg.nodes[adj]["label"] == ",":
                        or_found = True
                        break
                if not or_found:
                    op = ";"
                    dg.nodes[node]["label"] = ";"

            if op == ";" or op == "&":
                these_weights = sorted(these_weights, key=cmp_to_key(custom_key2))
                dg.nodes[node]["weights"] = these_weights
            elif op == ",":
                these_weights = sorted(
                    these_weights, key=cmp_to_key(custom_key2), reverse=True
                )
                dg.nodes[node]["weights"] = these_weights

            this_text = [r[1] for r in these_weights]
            first_atom_index = 0
            pattern = r"#\d+"
            for idx, txt in enumerate(this_text):
                if "," in txt:
                    txt = txt.split(",")[0]
                if ";" in txt:
                    txt = txt.split(";")[0]
                if txt in ATOMS or re.match(pattern, txt):
                    first_atom_index = idx
                    break

            # print(this_text, first_atom_index)
            this_text = moveToFront(this_text, first_atom_index)
            # print(this_text)
            dg.nodes[node]["weights"] = moveToFront(
                dg.nodes[node]["weights"], first_atom_index
            )

            op = dg.nodes[node]["label"]

            dg.nodes[node]["text"] = op.join(this_text)
            weights_in_order.append(these_weights)

    # print(dg.nodes)
    if atom_map != None and len(atom_map) > 0:
        return "[" + dg.nodes[0]["text"] + atom_map + "]", weights_in_order[-1], dg
    return "[" + dg.nodes[0]["text"] + "]", weights_in_order[-1], dg


def generate(
    test_smarts="[!a@H&D2;#7,#6;H;a-3;#7,!O,!#8&!O;#7,!O,!#8&!O++;*;H0]",
    title="figures/heatmaps/network.png",
    figsize=(3.6, 2),
):
    sanitized, group_smarts = sanitize_smarts_token(test_smarts)
    hmps, opss, x_tokss, dgs, titles = gen_data_structure(
        sanitized, group_smarts, test_smarts
    )

    fig = plt.figure(figsize=(3.5, 4), dpi=300)
    gs = gridspec.GridSpec(
        1,
        len(hmps),
        width_ratios=[len(hmps[i].T) + len(opss[i]) for i in range(len(hmps))],
        wspace=0.0,
        hspace=0.0,
        top=0.9,
        bottom=0.1,
        left=0.1,
        right=0.9,
    )

    ylim = 24
    for i, axs in enumerate(gs):
        axs = plt.subplot(gs[i])
        cmap = mcolors.ListedColormap(["black", "darkgrey", "lightgrey", "#f7ae1d"])
        bounds = [-15, -10, -6, -1, 0]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        hm = hmps[i].T
        hm = hm[:, :ylim]
        ops = opss[i][: len(opss[i]) - 1]
        hm2 = []
        ops2 = []
        for iix, ii in enumerate(hm):
            hm2.append(ii)
            ops2.append("")
            if iix < len(ops):
                if ops[iix] == "or (,)":
                    val = -5
                elif ops[iix] == "and (&)":
                    val = -10
                hm2.append([val] * len(ii))
                ops2.append(ops[iix])

        if i < len(hmps) - 1:
            hm2.append([-20] * len(hm[0]))
            ops2.append("and (;)")
        hm2 = np.array(hm2).T
        x_toks = list(x_tokss[i])[:ylim]
        if i == 0:
            axs.set_yticks(range(len(x_toks)), x_toks)
            axs.set_yticklabels(x_toks, fontsize=6, fontfamily="arial")
        else:
            axs.set_yticks([])
        axs.set_xticks(range(len(ops2)), ops2, rotation=90)
        axs.set_xticklabels(ops2, fontsize=6, rotation=90, fontfamily="arial")
        masked_array = np.ma.array(hm2, mask=np.isnan(hm2))
        # cmap = matplotlib.cm.plasma
        cmap.set_bad("beige", 1.0)
        axs.imshow(hm2, cmap=cmap, norm=norm)
        for idx1, ii in enumerate(hm2):
            for idx2, jj in enumerate(ii):
                if not np.isnan(jj) and jj > -1:
                    axs.text(
                        idx2,
                        idx1,
                        str(int(jj)),
                        ha="center",
                        va="center",
                        color="black",
                        fontsize=6,
                        fontfamily="arial",
                    )
        for i22 in range(hm2.shape[0]):
            axs.axhline(i22 - 0.5, color="black", linewidth=0.8)
        # axs.set_title(titles[i], fontsize=6, fontfamily='arial')
        axs.set_aspect("auto")

    plt.suptitle(test_smarts, fontsize=6, fontfamily="arial")
    plt.show()
    # plt.savefig(title+"-heatmap.png", dpi=300, bbox_inches='tight', pad_inches=0.01)

    plt.close()

    labels2 = {}
    node_list_tokens = []
    node_list_junctions = []
    for n in dgs.nodes():
        labels2[n] = dgs.nodes[n]["label"]
        if labels2[n] not in [";", ",", "&"]:
            node_list_tokens.append(n)
        else:
            node_list_junctions.append(n)

    plt.figure(figsize=figsize, dpi=300)

    pos = nx.nx_agraph.graphviz_layout(dgs, prog="dot")
    nx.draw_networkx_nodes(
        dgs, pos, nodelist=node_list_tokens, node_size=125, node_color="#f7ae1d"
    )
    nx.draw_networkx_nodes(
        dgs, pos, nodelist=node_list_junctions, node_size=50, node_color="grey"
    )
    nx.draw_networkx_labels(
        dgs, pos, labels2, font_size=6, font_family="arial", font_color="black"
    )
    nx.draw_networkx_edges(dgs, pos)

    # plt.savefig(title + "-tree.png", dpi=300, bbox_inches="tight", pad_inches=0.01)

    plt.show()
