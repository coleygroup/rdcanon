import hashlib
from collections import deque
import networkx as nx
from rdkit import Chem
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import gridspec
import numpy as np
from lark import Lark, Transformer
from rdcanon.askcos_prims import prims as prims1
from rdcanon.pubchem_prims import prims as prims2
from rdcanon.drugbank_prims_with_nots import prims as prims3
from rdcanon.np_prims import prims as prims4
import random
from functools import cmp_to_key


#PRIMITIVE:  "D" | "H" | "h" | "R" | "r" | "v" | "X" | "x" | "-" | "+" | "#" 
#                | "*" | "a" | "A" | "@" | "@@"
#                | "O" | "C" | "N" | "o" | "c" | "n" | "S" | "s" | "P" | "B" | "b" | "F" | "I" | "Cl" | "Br"
#                | "Se" | "Si" | "Sn" | "As" | "Te" | "Pb" | "Zn" | "Cu" | "Fe" | "Mg" | "Na" | "Ca" | "Al"
#                | "K" | "Li" | "Mn" | "Zr" | "Co" | "Ni" | "Cd" | "Ag" | "Au" | "Pt" | "Pd" | "Ru" | "Rh"
#                | "Ir" | "Ti" | "V" | "W" | "Mo" | "Hg" | "Tl" | "Bi" | "Ba" | "Sr" | "Cs" | "Rb" | "Be"


bond_value_map = {  "UNSPECIFIED": 1000,
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
                    "ZERO": 4
}

def hash_smarts(in_smarts, in_prims, func="sha256"):
    if func == 'sha256':
        smarts_bytes = in_smarts.encode()
        hasher = hashlib.sha256()
        hasher.update(smarts_bytes)
        val = int(hasher.hexdigest(), 16)
    else:
        if in_smarts[0] == "!":
            if in_smarts in in_prims:
                val = in_prims[in_smarts]
                # print(in_smarts, val)
            elif in_smarts[1:] not in in_prims:
                val = hash_smarts(in_smarts[1:], {})/1e78
            else:
                val = 1/in_prims[in_smarts[1:]]
        else:
            if in_smarts not in in_prims:
                val = hash_smarts(in_smarts, {})/1e78
            else:
                val = in_prims[in_smarts]
    # print(val)
    return val


# print(embedding, prims)
prims1["*"] = 10e64
prims2["*"] = 10e64
prims3["*"] = 10e64
prims4["*"] = 10e64

for k in prims1: prims1[k] = prims1[k] + hash_smarts(k, {})/1e78
for k in prims2: prims2[k] = prims2[k] + hash_smarts(k, {})/1e78
for k in prims3: prims3[k] = prims3[k] + hash_smarts(k, {})/1e78
for k in prims4: prims4[k] = prims4[k] + hash_smarts(k, {})/1e78

labels = ["!", \
          "*", "a", "A", "@", "@@", \
          "C", "N", "O", "o", "c", "n", "s", "S", "P", "p", "B", "b", "F", "I", "Cl", "Br", \
          "Se", "Si", "Sn", "As", "Te", "Pb", "Zn", "Cu", "Fe", "Mg", "Na", "Ca", "Al", \
          "K", "Li", "Mn", "Zr", "Co", "Ni", "Cd", "Ag", "Au", "Pt", "Pd", "Ru", "Rh", \
          "Ir", "Ti", "V", "W", "Mo", "Hg", "Tl", "Bi", "Ba", "Sr", "Cs", "Rb", "Be", "se", "te"]

ATOMS = [ "a", "A",
          "C", "N", "O", "o", "c", "n", "s", "S", "P", "p", "B", "b", "F", "I", "Cl", "Br", \
          "Se", "Si", "Sn", "As", "Te", "Pb", "Zn", "Cu", "Fe", "Mg", "Na", "Ca", "Al", \
          "K", "Li", "Mn", "Zr", "Co", "Ni", "Cd", "Ag", "Au", "Pt", "Pd", "Ru", "Rh", \
          "Ir", "Ti", "V", "W", "Mo", "Hg", "Tl", "Bi", "Ba", "Sr", "Cs", "Rb", "Be", "se", "te"]


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
            if cur in ["rec_start", "brack_start", "rec_end", "brack_end", "&", ","]:
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
            if cur[0] == "nested":
                stack.appendleft('rec_start')
            if cur[0] == "bracketed":
                stack.appendleft('brack_start')
            for r in cur[1]:
                stack.appendleft(r)
            if cur[0] == "bracketed":
                stack.appendleft('brack_end')
            if cur[0] == "nested":
                stack.appendleft('rec_end')


        tok = ""
        for r in reversed(seq):
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
            if r in ["=", "-", "#", "~", ":", ".", "@", "/", "!", "&", ","]:
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

                result = {'!': -1, 
                        "D": -1, "H": -1, "h": -1, "R": -1, "r": -1, "v": -1, "X": -1, "x": -1, "-": -1, "+": -1, "#": -1,
                        "*": -1, "a": -1, "A": -1, "@": -1, "@@": -1,
                        "bond": -1, "rec": -1,
                        "C": -1, "N": -1, "O": -1, "o":-1, "c":-1, "n":-1,  "S":-1, "s":-1, "P":-1, "p":-1, "B": -1, "b": -1, "F": -1, "I": -1, "Cl": -1, "Br": -1, 
                        "Se": -1, "Si": -1, "Sn": -1, "As": -1, "Te": -1, "Pb": -1, "Zn": -1, "Cu": -1, "Fe": -1, "Mg": -1, "Na": -1, "Ca": -1, "Al": -1,
                        "K": -1, "Li": -1, "Mn": -1, "Zr": -1, "Co": -1, "Ni": -1, "Cd": -1, "Ag": -1, "Au": -1, "Pt": -1, "Pd": -1, "Ru": -1, "Rh": -1,
                        "Ir": -1, "Ti": -1, "V": -1, "W": -1, "Mo": -1, "Hg": -1, "Tl": -1, "Bi": -1, "Ba": -1, "Sr": -1, "Cs": -1, "Rb": -1, "Be": -1, "se": -1, "te": -1
                        }

                if len(r) == 2:
                    result["bond"] = str(r[0][1])
                    atms = r[1][1]
                else:
                    atms = r[0][1]

                deg_found = False
                prev_prim = False
                for rr in atms:
                    if rr[0] == '!':
                        result['!'] = True
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
            result = {'!': -1, 
                "D": -1, "H": -1, "h": -1, "R": -1, "r": -1, "v": -1, "X": -1, "x": -1, "-": -1, "+": -1, "#": -1,
                "*": -1, "a": -1, "A": -1, "@": -1, "@@": -1,
                "bond": -1, "rec": 1,
                "C": -1, "N": -1, "O": -1, "o":-1, "c":-1, "n":-1,  "S":-1, "s":-1, "P":-1, "p":-1, "B": -1, "b": -1, "F": -1, "I": -1, "Cl": -1, "Br": -1, 
                "Se": -1, "Si": -1, "Sn": -1, "As": -1, "Te": -1, "Pb": -1, "Zn": -1, "Cu": -1, "Fe": -1, "Mg": -1, "Na": -1, "Ca": -1, "Al": -1,
                "K": -1, "Li": -1, "Mn": -1, "Zr": -1, "Co": -1, "Ni": -1, "Cd": -1, "Ag": -1, "Au": -1, "Pt": -1, "Pd": -1, "Ru": -1, "Rh": -1,
                "Ir": -1, "Ti": -1, "V": -1, "W": -1, "Mo": -1, "Hg": -1, "Tl": -1, "Bi": -1, "Ba": -1, "Sr": -1, "Cs": -1, "Rb": -1, "Be": -1, "se": -1, "te": -1
            }
            results.append(result)

        return results, tok

    def not1(self, args):
        return "!", args[0]  # Returning '!' to indicate its presence

    def not2(self, args):
        return "!", args  # Returning '!' to indicate its presence
    
    def symbol(self, args):
        return "prim", args[0]
    
    def symbol_single(self, args):
        return "prim", args[0]

    def degree1(self, args):
        return 'deg', args[0]

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
 

def split_smarts(input_smarts):
    tsmarts = Chem.MolFromSmarts(input_smarts)
    out_sm = Chem.MolToSmarts(tsmarts)
    # out_sm = input_smarts
    if out_sm[0] == "[" and out_sm[-1] == "]":
        deb = out_sm[1:-1]
    else:
        deb = out_sm
    debsp = custom_split(deb, ";")
    return debsp


def custom_split(input_string, delimiter=';', nested_start='$(', nested_end=')'):
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
        elif input_string.startswith("(", i) and not input_string.startswith(nested_start, i-1):
            nested_level += 1
            current.append(input_string[i])
        elif input_string.startswith(nested_end, i) and nested_level:
            nested_level -= 1
            current.append(input_string[i])
        elif input_string[i] == delimiter and nested_level == 0:
            parts.append(''.join(current))
            current = []
        else:
            current.append(input_string[i])
        i += 1
    if current:
        parts.append(''.join(current))
    return parts


def categorize_string(sm, groups, nested_start='$(', nested_end=')'):
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
    key = ''.join(sorted(found_chars)) if found_chars else 'p'
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
        "p": []
    }
    for sm in split_smarts:
        categorize_string(sm, groups)
    return groups


def check_special_chars_outside_nested(sm, nested_start='$(', nested_end=')'):
    """
    Check if "!", "&", and "," are in the string `sm`, but not inside `$( x )` structures.
    """
    inside_nested = False
    found_chars = {"&": False}
    i = 0
    while i < len(sm):
        if sm.startswith(nested_start, i):
            inside_nested = True
            i += len(nested_start) - 1  # Adjust index to skip nested_start
        elif sm.startswith(nested_end, i) and inside_nested:
            inside_nested = False
            i += len(nested_end) - 1  # Adjust index to skip nested_end

        if not inside_nested:
            if sm[i] in found_chars:
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
        # print(token)
        prim_tree = parser.parse(token)
        tokens.append((prim_tree,))
    return ('and', tokens)


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
            # print(token)
            prim_tree = parser.parse(token)
            tokens.append((prim_tree,))
    return ('or', tokens)


def reorder_internal_split_smarts(grouped_split_smarts):
    ordered_out = {}
    for group in all_groups:
        ordered_out[group] = []
    for group in all_groups:
        # print(group, grouped_split_smarts[group])
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
                # print(smt)
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
                # print(smt)
                prim_tree = parser.parse(smt)
                ordered_out[group].append((prim_tree,))
    return ordered_out


def sanitize_smarts_token(in_token):
    o1 = list(set(split_smarts(in_token)))
    # print(o1)
    o2 = group_split_smarts(o1)
    # print(o2)
    o3 = reorder_internal_split_smarts(o2)
    return o3, o2


def parse_label(lab):
    tx = ""
    if lab["rec"] == 1:
        return "rec"
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


def invert_nested_values(nested_obj):
    """
    Recursively invert the values in a nested structure of lists and tuples.

    Parameters:
    nested_obj (list/tuple): A nested object consisting of lists and tuples.

    Returns:
    list/tuple: A new nested object with each value being 1/original_value.
    """
    if isinstance(nested_obj, (list, tuple)):
        return type(nested_obj)(invert_nested_values(x) for x in nested_obj)
    else:
        return 1/nested_obj
    

amol = Chem.MolFromSmarts("C:C")
abond = amol.GetBondWithIdx(0)

smol = Chem.MolFromSmarts("C-C")
sbond = smol.GetBondWithIdx(0)


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
            for iidx,i in enumerate(cur[0][1]):
                if iidx != len(cur[0][1])-1:
                    ops.append("and (&)")
                stack.append((i, n))
        elif cur[0][0] == "or":
            label = ","
            for iidx, i in enumerate(cur[0][1]):
                stack.append((i,n))
                if iidx != len(cur[0][1])-1:
                    ops.append("or (,)")
        else:
            # print(cur[0][0])
            result, tok = transformer.transform(cur[0][0])
            # print(result)
            # print("results", len(result))
            # for wow in result:
                # print(wow)
            label = parse_label(result[0])
            if label == "rec":
                if tok[0] == "!":
                    inc = 1
                else:
                    inc = 0
                # print("recursive", tok)
                mm_rec = Chem.MolFromSmarts(tok[2+inc:-1])
                for atom_idx,atom in enumerate(mm_rec.GetAtoms()):
                    # print(atom.GetSmarts())
                    sm, scs, wts = order_token_canon(atom.GetSmarts(), None, prims)
                    weights.append(scs)
                    atom.SetQuery(Chem.MolFromSmarts(sm).GetAtoms()[0])
                    if atom_idx + 1 < mm_rec.GetNumAtoms():

                        bond = mm_rec.GetBondBetweenAtoms(atom_idx, atom_idx+1)

                        if bond:
                            if bond.Match(abond) and bond.Match(sbond):
                                bondv = bond_value_map["UNSPECIFIED"]

                            else:
                                bondv = bond_value_map[bond.GetBondType().name]
                        else:
                            bondv = 2
                    else:
                        bondv = 2
                    weights.append([bondv])
                if inc:
                    # weights = invert_nested_values(weights)
                    label = "!$(" + Chem.MolToSmarts(mm_rec) + ")"
                else:
                    label = "$(" + Chem.MolToSmarts(mm_rec) + ")"
                # print(tok, label, tuple(weights))
                # print(label)
                
            # print(label)
                
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


def gen_data_structure(sanitized, group_smarts, test_smarts, prims='askcos'):
    trees = []
    titles = []

    token_num = 1
    for ixiii, r in enumerate(sanitized): # group
        if len(sanitized[r]) == 0:
            continue
        for ixi, i in enumerate(sanitized[r]): # sample
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

# Define the grammar
grammar = """
start: "[" item* "]"

item: (open item* close | not2? nested_rule | component | bond_symbol nested_rule | bracketed_rule | bond_symbol | degree1 ) operator_symbol?

open: "("
close: ")"

component: (bond_symbol atom) | atom

atom: not1? symbol degree1? 

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

symbol_single: PRIMITIVE_SINGLE
                
PRIMITIVE_SINGLE:  "*" | "a" | "A" | "@" | "@@"
                    | "O" | "C" | "N" | "o" | "c" | "n" | "S" | "s" | "P" | "p" | "B" | "b" | "F" | "I" | "Cl" | "Br"
                    | "Se" | "Si" | "Sn" | "As" | "Te" | "Pb" | "Zn" | "Cu" | "Fe" | "Mg" | "Na" | "Ca" | "Al"
                    | "K" | "Li" | "Mn" | "Zr" | "Co" | "Ni" | "Cd" | "Ag" | "Au" | "Pt" | "Pd" | "Ru" | "Rh"
                    | "Ir" | "Ti" | "V" | "W" | "Mo" | "Hg" | "Tl" | "Bi" | "Ba" | "Sr" | "Cs" | "Rb" | "Be" | "se" | "te"

                    
PRIMITIVE_MULTI:  "D" | "H" | "h" | "R" | "r" | "v" | "X" | "x" | "-" | "+" | "#" 

symbol_multi: PRIMITIVE_MULTI

bond_symbol: BOND_PRIMITIVE

BOND_PRIMITIVE: "-" | "=" | "#" | "~" | ":" | "." | "@" | "/"

operator_symbol: OPERATOR_PRIMITIVE

OPERATOR_PRIMITIVE: ";" | "," | "&"

degree1: INTEGER

INTEGER: DIGIT+

DIGIT: "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9"

%ignore " "
"""

parser = Lark(grammar, parser='lalr')
transformer = SMARTSTransformer()


def flatten_list(nested_list):
    """ Flatten a nested list into a single list. """
    result = []
    if isinstance(nested_list, list) or isinstance(nested_list, tuple):
        for item in nested_list:
            result.extend(flatten_list(item))
    else:
        result.append(nested_list)
    return result

def recursive_compare(list1, list2):
    index1 = index2 = 0

    # print("-")
    # print(list1)
    # print(list2)
    # print("-")

    if not list2 or not list1:
        return True

    while index1 < len(list1) and index2 < len(list2):
        sub_item1, sub_item2 = list1[index1], list2[index2]
        # print("comparing")
        # print(sub_item1)
        # print(sub_item2)
        # print()
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


def swapPositions(list, pos1, pos2):
     
    list[pos1], list[pos2] = list[pos2], list[pos1]
    return list

def order_token_canon(in_smarts_token="[!a@H&D2;#7,#6;H;a-3;#7,!O,!#8&!O;#7,!O,!#8&!O++;*;H0]", atom_map=None, embedding="drugbank"):
    # print(in_smarts_token)

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
            raise ValueError("embedding must be 'askcos', 'pubchem', 'drugbank', 'npatlas', or a dictionary of primitives")

    # print(in_smarts_token)

    sanitized, group_smarts = sanitize_smarts_token(in_smarts_token)
    # print(group_smarts)
    hmps, opss, x_tokss, dg, titles = gen_data_structure(sanitized, group_smarts, in_smarts_token, prims)

    # print(in_smarts_token, dg.nodes())


    # initialize weights
    remaining_nodes = []
    for node in dg.nodes():
        if dg.in_degree(node) == 0:
            prim_sm = dg.nodes[node]['label']
            if prim_sm[0] == "!":
                inc = 1
            else:
                inc = 0
            if prim_sm[0 + inc] == "$":
                # new_trees = hash_smarts(prim_sm, prims, func='embedded')
                # for tree in new_trees:
                    # ordered_weights = get_tree_traversal(tree)
                # dg.nodes[node]['weights'] = hash_smarts(prim_sm, prims, func='embedded')
                # print(prim_sm, dg.nodes[node]['weights'])
                pass
            else:
                # dg.nodes[node]['weight'] = hash_smarts(prim_sm, prims, func='embedded')
                # print(prim_sm, dg.nodes[node]['weight'])
                dg.nodes[node]['weights'] = [hash_smarts(prim_sm, prims, func='embedded')]
                # print(prim_sm, dg.nodes[node]['weights'])
            # print(prim_sm, dg.nodes[node]['weights'])

            # print(prim_sm)
            # print(dg.nodes[node]['weights'])
            # print()
            dg.nodes[node]['text'] = prim_sm
        else:
            remaining_nodes.append(node)
    stack = deque()
    stack.append(0)

    weights_in_order = []
    been_sorted = []
    while stack:
        node = stack.popleft()
        if node in been_sorted:
            continue
        these_weights = []
        recurse = False

        op = dg.nodes[node]['label']
        for neighbor in dg.in_edges(node):
            adj = neighbor[0]
            if 'weights' not in dg.nodes[adj]:
                stack.append(adj)
                stack.append(node)
                recurse = True
                break
            else:
                if "text" in dg.nodes[adj]:
                    txt = dg.nodes[adj]["text"]
                else:
                    txt = None
                these_weights.append((dg.nodes[adj]['weights'],txt))

        if not recurse:
            been_sorted.append(node)

            if op == "&":
                or_found = False
                for neighbor in dg.out_edges(node):
                    adj = neighbor[1]
                    if dg.nodes[adj]['label'] == ",":
                        or_found = True
                        break
                if not or_found:
                    op = ";"
                    dg.nodes[node]['label'] = ";"

            # print("ordering", node, dg.nodes[node]['label'], these_weights)
            if op == ";" or op == "&":
                # dg.nodes[node]['weight'] = np.min(these_weights)
                # print(these_weights)
                # print(len(these_weights))
                # print(flatten_list(these_weights))
                these_weights = sorted(these_weights, key=cmp_to_key(custom_key2))
                # print(these_weights)
                # print()
                # print(these_weights)
                dg.nodes[node]['weights'] = these_weights
            elif op == ",":
                # dg.nodes[node]['weight'] = np.max(these_weights)
                these_weights = sorted(these_weights, key=cmp_to_key(custom_key2), reverse=True)
                dg.nodes[node]['weights'] = these_weights



            # print(these_weights)
            this_text = [r[1] for r in these_weights]
            first_atom_index = 0
            for idx, txt in enumerate(this_text):
                if txt in ATOMS:
                    first_atom_index = idx
                    break


            # print(this_text, 0, first_atom_index)
            this_text = swapPositions(this_text, 0, first_atom_index)
            # print(this_text, 0, first_atom_index)
            dg.nodes[node]['weights'] = swapPositions(dg.nodes[node]['weights'], 0, first_atom_index)
            # if these_weights[0][1][0] == "-" or these_weights[0][1][0] == "+":
            #     dg.nodes[node]['weights'] = these_weights[1:]
            #     dg.nodes[node]['weights'].insert(1, these_weights[0])
            #     this_text = [r[1] for r in these_weights[1:]]
            #     this_text.insert(1, these_weights[0][1])


            op = dg.nodes[node]['label']

            dg.nodes[node]['text'] = op.join(this_text)
            weights_in_order.append(these_weights)

                # print(these_weights)

            # print("ordered", node, dg.nodes[node]['label'], these_weights)
            # print()

    # stack = deque()
    # stack.append(0)

    # weights_in_order = []

    # while stack:
    #     n_in = stack.popleft()
    #     node = n_in

    #     weights = []
    #     # print(node, dg.nodes[node]['label'])
    #     for neighbor in dg.in_edges(node):
    #         adj = neighbor[0]
    #         # print(dg.nodes[adj]['weights'])
    #         weights.append((dg.nodes[adj]['weights'], adj))
    #     # print(weights)
    #     op = dg.nodes[node]['label']
    #     if op == ";":
    #         weights = sorted(weights, key=cmp_to_key(custom_key))
    #     elif op == ",":
    #         weights = sorted(weights, key=cmp_to_key(custom_key), reverse=True)
    #     elif op == "&":
    #         weights = sorted(weights, key=cmp_to_key(custom_key))

    #     # print(weights)

    #     recurse = False
    #     this_text = []
    #     these_weights = []
    #     for w in weights:
    #         adj = w[1]
    #         if 'text' not in dg.nodes[adj]:
    #             stack.append(adj)
    #             stack.append(node)
    #             recurse = True
    #             break
    #         else:
    #             this_text.append(dg.nodes[adj]['text'])
    #             these_weights.append(dg.nodes[adj]['weights'])
    #     if not recurse:
    #         # print("writing", node, dg.nodes[node]['label'], this_text)
    #         op = dg.nodes[node]['label']
    #         # dg.nodes[node]['weights'] = weights[0]
    #         dg.nodes[node]['text'] = op.join(this_text)
    #         weights_in_order.append(these_weights)

    # print(dg.nodes[0]['text'])

    # print(weights_in_order[-1])
    # if type(weights_in_order[-1]) == float:
        # weights = weights_in_order[-1]
    # else:
        # weights = []
        # for rr in weights_in_order[-1]:
            # weights.append(rr)
    # for rr in dg.nodes():
        # print(dg.nodes[rr]['text'], dg.nodes[rr]['weight'])
    # print(weights)
    if atom_map != None and len(atom_map) > 0:
        return "[" + dg.nodes[0]['text'] + atom_map + "]", weights_in_order[-1], dg
    return "[" + dg.nodes[0]['text'] + "]", weights_in_order[-1], dg


def generate(test_smarts = "[!a@H&D2;#7,#6;H;a-3;#7,!O,!#8&!O;#7,!O,!#8&!O++;*;H0]", title="figures/heatmaps/network.png", figsize=(3.6,2)):

    sanitized, group_smarts = sanitize_smarts_token(test_smarts)
    hmps, opss, x_tokss, dgs, titles = gen_data_structure(sanitized, group_smarts, test_smarts)


    fig = plt.figure(figsize=(3.5, 4), dpi=300) 
    gs = gridspec.GridSpec(1, len(hmps), width_ratios=[len(hmps[i].T)+len(opss[i]) for i in range(len(hmps))],
            wspace=0.0, hspace=0.0, top=0.9, bottom=0.1, left=0.1, right=0.9) 

    ylim = 24
    for i,axs in enumerate(gs):

        axs = plt.subplot(gs[i])
        cmap = mcolors.ListedColormap(["black", "darkgrey","lightgrey", '#f7ae1d'])
        bounds = [-15,-10, -6, -1, 0]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        hm = hmps[i].T
        hm = hm[:,:ylim]
        ops = opss[i][:len(opss[i])-1]
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
                hm2.append([val]*len(ii))
                ops2.append(ops[iix])

        if i < len(hmps)-1:
            hm2.append([-20]*len(hm[0]))
            ops2.append("and (;)")
        hm2 = np.array(hm2).T
        x_toks = list(x_tokss[i])[:ylim]
        if i == 0:
            axs.set_yticks(range(len(x_toks)), x_toks)
            axs.set_yticklabels(x_toks, fontsize=6,fontfamily='arial')
        else:
            axs.set_yticks([])
        axs.set_xticks(range(len(ops2)), ops2, rotation=90)
        axs.set_xticklabels(ops2, fontsize=6, rotation=90, fontfamily='arial')
        masked_array = np.ma.array(hm2, mask=np.isnan(hm2))
        # cmap = matplotlib.cm.plasma
        cmap.set_bad('beige',1.)
        axs.imshow(hm2, cmap=cmap, norm=norm)
        for idx1, ii in enumerate(hm2):
            for idx2,jj in enumerate(ii):
                if not np.isnan(jj) and jj > -1:
                    axs.text(idx2, idx1, str(int(jj)), ha='center', va='center', color='black', fontsize=6, fontfamily='arial')
        for i22 in range(hm2.shape[0]):
            axs.axhline(i22 - 0.5, color='black', linewidth=0.8)
        # axs.set_title(titles[i], fontsize=6, fontfamily='arial')
        axs.set_aspect("auto")

    # plt.suptitle(test_smarts, fontsize=6, fontfamily='arial')
    plt.show()
    # plt.savefig(title+"-heatmap.png", dpi=300, bbox_inches='tight', pad_inches=0.01)

    plt.close()

    labels2 = {}
    node_list_tokens = []
    node_list_junctions = []
    for n in dgs.nodes():
        labels2[n] = dgs.nodes[n]['label']
        if labels2[n] not in [";",",","&"]:
            node_list_tokens.append(n)
        else:
            node_list_junctions.append(n)

    plt.figure(figsize=figsize, dpi=300)

    pos = nx.nx_agraph.graphviz_layout(dgs, prog='dot')
    nx.draw_networkx_nodes(dgs, pos, nodelist=node_list_tokens, node_size=125, node_color='#f7ae1d')
    nx.draw_networkx_nodes(dgs, pos, nodelist=node_list_junctions, node_size=50, node_color='grey')
    nx.draw_networkx_labels(dgs, pos, labels2, font_size=6, font_family='arial', font_color='black')
    nx.draw_networkx_edges(dgs, pos)

    plt.savefig(title+"-tree.png", dpi=300, bbox_inches='tight', pad_inches=0.01)

    # plt.show()
