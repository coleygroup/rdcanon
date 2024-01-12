from rdcanon.main import canon_smarts, canon_reaction_smarts, random_smarts
from rdkit import Chem
from rdkit.Chem import AllChem
import timeit
import time
from matplotlib import pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np
import pandas as pd
import os


def test_two_atom_smarts():
    s_test1 = "[C;H0;+0]-[C;H1;+0]"
    s_test2 = "[C;H1;+0]-[C;H0;+0]"
    assert canon_smarts(s_test1) == canon_smarts(s_test2)

    s_test1 = "[C;H0;+0]=[C;H1;+0]"
    s_test2 = "[C;H1;+0]=[C;H0;+0]"
    assert canon_smarts(s_test1) == canon_smarts(s_test2)

    s_test1 = "[C;H0;++]=[C;H1;+0]"
    s_test2 = "[C;H1;+0]=[C;H0;+2]"
    assert canon_smarts(s_test1) == canon_smarts(s_test2)

    s_test1 = "[C;H1;+0]-[C;H1;+0]"
    canon_smarts(s_test1)


def test_permutation_of_monosubstituted_benzene():
    results = []
    s_test1 = "[Br;!Cl;H10;X10][c;H0]1[c;H0][c;H0][c;H0][c;H0][c;H0]1"
    s_test2 = "[c;H0]([Br;!Cl;H10;X10])1[c;H0][c;H0][c;H0][c;H0][c;H0]1"
    s_test3 = "[c;H0]1[c;H0]([Br;!Cl;H10;X10])[c;H0][c;H0][c;H0][c;H0]1"
    s_test4 = "[c;H0]1[c;H0][c;H0]([Br;!Cl;H10;X10])[c;H0][c;H0][c;H0]1"
    s_test5 = "[c;H0]1[c;H0][c;H0][c;H0]([Br;!Cl;H10;X10])[c;H0][c;H0]1"
    s_test6 = "[c;H0]1[c;H0][c;H0][c;H0][c;H0]([Br;!Cl;H10;X10])[c;H0]1"
    s_test7 = "[c;H0]1[c;H0][c;H0][c;H0][c;H0][c;H0]([Br;!Cl;H10;X10])1"
    results.append(canon_smarts(s_test1))
    results.append(canon_smarts(s_test2))
    results.append(canon_smarts(s_test3))
    results.append(canon_smarts(s_test4))
    results.append(canon_smarts(s_test5))
    results.append(canon_smarts(s_test6))
    results.append(canon_smarts(s_test7))
    assert len(set(results)) == 1


def test_symmetric_molecules():
    s_test1 = "[Cl][C][C][C][C][C][C][N][C][C][C][C][C][C][Br]"
    s_test2 = "[Br][C][C][C][C][C][C][N][C][C][C][C][C][C][Cl]"
    assert canon_smarts(s_test1) == canon_smarts(s_test2)

    s_test1 = "[Br][C][C][C][C][C][C][N][C][C][C][C][C][C]=[Br]"
    s_test2 = "[Br]=[C][C][C][C][C][C][N][C][C][C][C][C][C][Br]"
    assert canon_smarts(s_test1) == canon_smarts(s_test2)

    s_test1 = "[Cl][C][C][C][C][C][C][C][N][N][C][C][C][C][C][C][C][Br]"
    s_test2 = "[Br][C][C][C][C][C][C][C][N][N][C][C][C][C][C][C][C][Cl]"
    assert canon_smarts(s_test1) == canon_smarts(s_test2)

    s_test1 = "Clc1ccc(CCCCCCCNNCCCCCCCc2ccc(Br)cc2)cc1"
    s_test2 = "Brc1ccc(CCCCCCCNNCCCCCCCc2ccc(Cl)cc2)cc1"
    assert canon_smarts(s_test1) == canon_smarts(s_test2)

    s_test1 = "BrCCCCCCN(CCCCCCI)CCCCCCCl"
    s_test2 = "BrCCCCCCN(CCCCCCCl)CCCCCCI"
    s_test3 = "ClCCCCCCN(CCCCCCBr)CCCCCCI"
    s_test4 = "ClCCCCCCN(CCCCCCI)CCCCCCBr"
    s_test5 = "ICCCCCCN(CCCCCCBr)CCCCCCCl"
    s_test6 = "ICCCCCCN(CCCCCCCl)CCCCCCBr"
    assert (
        canon_smarts(s_test1)
        == canon_smarts(s_test2)
        == canon_smarts(s_test3)
        == canon_smarts(s_test4)
        == canon_smarts(s_test5)
        == canon_smarts(s_test6)
    )


def atom_invariant_permutations():
    s_test1 = "[Br;!Cl;H10;X10][c;H0]1[c;H0][c;H0][c;H0][c;H0][c;H0]1"
    s_test2 = "[!Cl;H10;X10;Br][c;H0]1[c;H0][c;H0][c;H0][c;H0][c;H0]1"
    s_test3 = "[H10;X10;Br;!Cl][c;H0]1[c;H0][c;H0][c;H0][c;H0][c;H0]1"
    s_test4 = "[X10;Br;!Cl;H10][c;H0]1[c;H0][c;H0][c;H0][c;H0][c;H0]1"
    assert (
        canon_smarts(s_test1)
        == canon_smarts(s_test2)
        == canon_smarts(s_test3)
        == canon_smarts(s_test4)
    )

    s_test1 = "[Br;Cl,B,C]"
    s_test2 = "[Br;B,C,Cl]"
    s_test3 = "[Br;C,Cl,B]"
    s_test4 = "[Cl,B,C;Br]"
    s_test5 = "[B,C,Cl;Br]"
    s_test6 = "[C,Cl,B;Br]"
    assert (
        canon_smarts(s_test1)
        == canon_smarts(s_test2)
        == canon_smarts(s_test3)
        == canon_smarts(s_test4)
        == canon_smarts(s_test5)
        == canon_smarts(s_test6)
    )


def stereochemistry_permutations():
    s_test1 = "[C][C@][O]"
    s_test2 = "[C][C@@][O]"
    assert canon_smarts(s_test1) != canon_smarts(s_test2)

    isomers = [
        "I-[C@@](-Br)(-Cl)-C-[C@](-P)(-O)-B",
        "I-[C@@](-Br)(-Cl)-C-[C@@](-P)(-B)-O",
        "I-[C@](-Cl)(-Br)-C-[C@](-P)(-O)-B",
        "I-[C@](-Cl)(-Br)-C-[C@@](-O)(-P)-B",
        "O-[C@](-C-[C@](-I)(-Br)-Cl)(-P)-B",
        "O-[C@](-C-[C@@](-Br)(-I)-Cl)(-P)-B",
        "O-[C@@](-P)(-C-[C@](-I)(-Br)-Cl)-B",
        "O-[C@@](-P)(-C-[C@@](-Br)(-I)-Cl)-B",
        "P-[C@](-C-[C@](-Cl)(-I)-Br)(-B)-O",
        "P-[C@](-C-[C@@](-I)(-Cl)-Br)(-B)-O",
        "P-[C@@](-B)(-C-[C@](-Cl)(-I)-Br)-O",
        "P-[C@@](-B)(-C-[C@@](-I)(-Cl)-Br)-O",
        "B-[C@](-P)(-C-[C@@](-Br)(-I)-Cl)-O",
        "C(-[C@](-B)(-P)-O)-[C@@](-I)(-Cl)-Br",
        "B-[C@](-P)(-C-[C@](-Br)(-Cl)-I)-O",
        "C(-[C@@](-Cl)(-Br)-I)-[C@](-O)(-B)-P",
    ]
    assert len(set([canon_smarts(s) for s in isomers])) == 1

    isomers = [
        "C(=C/O)\C-C(=C-[C@@](-Br)(-Cl)-[H1@&C](-B)(-P)-O)-C",
        "C(\O)=C/C-C(=C-[C@@](-[C@@&H1](-B)(-O)-P)(-Br)-Cl)-C",
        "Br-[C@](-Cl)(-[H1@@&C](-B)(-O)-P)-C=C(-C)-C/C=C\O",
        "C(-C)(-C/C=C\O)=C-[C@](-Br)(-[C@&H1](-B)(-P)-O)-Cl",
        "O/C=C\C-C(=C-[C@](-[H1@&C](-B)(-P)-O)(-Cl)-Br)-C",
        "C(=C-[C@](-Br)(-[H1@@&C](-B)(-O)-P)-Cl)(-C/C=C\O)-C",
        "B-[H1@&C](-[C@](-Br)(-Cl)-C=C(-C)-C/C=C\O)(-O)-P",
    ]
    assert len(set([canon_smarts(s) for s in isomers])) == 1


def test_recursive():
    s_test1 = "[$([C&X2,N&X2](=[$([C])])=[C])]"
    assert canon_smarts(s_test1) == canon_smarts(canon_smarts(s_test1))
    s_test1 = "[$([NX2]=[NX2+]=[NX1-]),$([NX2]=[NX2+]=N),$([NX2-]-[NX2+]#[NX1]),$([NX3](=[OX1])=[OX1]),$([NX3+](=[OX1])[O-]),$([NX2+]#[CX1-]),$([NX2]#[CX1]),$([OX1-,OH1][#7X4+]([*])([*])([*])),$([OX1]=[#7X4v5]([*])([*])([*])),$([OX1-,OH1][#7X3+R](~[R])(~[R])),$([OX1]=[#7v5X3R](~[R])(~[R])),$([*+]~[*-])]"
    s_test2 = "[$([N&X3&+](=[O&X1])[O&-]),$([N&X2]=[N&X2&+]=N),$([N&X2]=[N&X2&+]=[N&X1&-]),$([O&X1&-,O&H1][#7&X3&+&R](~[R])~[R]),$([N&X2]#[C&X1]),$([O&X1]=[#7&X4&v5](*)(*)*),$([O&X1&-,O&H1][#7&X4&+](*)(*)*),$([N&X3](=[O&X1])=[O&X1]),$([N&X2&+]#[C&X1&-]),$([+]~[-]),$([O&X1]=[#7&v5&X3&R](~[R])~[R]),$([N&X2&-]-[N&X2&+]#[N&X1])]"
    assert canon_smarts(s_test1) == canon_smarts(s_test2)
    s_test1 = "[#6&X3&H1]1-[#6&X3](:[$([#7&X3&H1&+,#7&X2&H0&+0]:[#6&X3&H1]:[#7&X3&H1]),$([#7&X3&H1])]:[#6&X3&H1]:[$([#7&X3&H1&+,#7&X2&H0&+0]:[#6&X3&H1]:[#7&X3&H1]),$([#7&X3&H1])]:1)-[X4&H2&C]"
    assert canon_smarts(s_test1)
    s_test1 = "[$([c,o][c,n]),$([n,c][c,o])]"
    s_test2 = "[$([n,c][c,o]),$([c,o][c,n])]"
    assert canon_smarts(s_test1) == canon_smarts(s_test2)
    s_test1 = "[$([n,c][c,o]),!$([c,o][c,n])]"
    s_test2 = "[!$([c,o][c,n]),$([n,c][c,o])]"
    assert canon_smarts(s_test1) == canon_smarts(s_test2)
    s_test1 = "[$([n,c][c,o])&!$([c]),!$([c,o][c,n])]"
    s_test2 = "[!$([c,o][c,n]),$([n,c][c,o])&!$([c])]"
    assert canon_smarts(s_test1) == canon_smarts(s_test2)
    s_test1 = "[$([n,c][c,o])&!$([c]),!$([c,o][c,n]);$([c,o])]"
    s_test2 = "[!$([c,o][c,n]),$([n,c][c,o])&!$([c]);$([c,o])]"
    s_test3 = "[$([c,o]);!$([c,o][c,n]),$([n,c][c,o])&!$([c])]"
    assert canon_smarts(s_test1) == canon_smarts(s_test2) == canon_smarts(s_test3)
    s_test1 = "[$([n,c][c,o])&!$([c]),!$([c,o][c,n]);$([$([c,o]),$([c])])]"
    s_test2 = "[$([n,c][c,o])&!$([c]),!$([c,o][c,n]);$([$([c]),$([c,o])])]"
    assert canon_smarts(s_test1) == canon_smarts(s_test2)


def test_permute_reactants():
    rxn1 = "([*:1]-[N&H0&+0:2](-[*:3])-C.[*:4]-[N&H0&+0:5](-[*:6])-C)>>([*:1]-[N&H1&+0:2]-[*:3].[*:4]-[N&H1&+0:5]-[*:6])"
    rxn2 = "([*:4]-[N&H0&+0:5](-[*:6])-C.[*:1]-[N&H0&+0:2](-[*:3])-C)>>([*:1]-[N&H1&+0:2]-[*:3].[*:4]-[N&H1&+0:5]-[*:6])"
    rxn3 = (
        "([N&H0&+0](-C)(-*)-*.[N&H0&+0](-C)(-*)-*)>>([N&H1&+0](-*)-*.[N&H1&+0](-*)-*)"
    )

    assert canon_reaction_smarts(rxn1, True) != canon_reaction_smarts(rxn2, True)
    assert (
        canon_reaction_smarts(rxn1)
        == canon_reaction_smarts(rxn2)
        == canon_reaction_smarts(rxn3)
    )

    rxn1 = "[*:1]-[N&H0&+0:2](-[*:3])-C.[*:4]-[N&H1&+0:5](-[*:6])-C>>([*:1]-[N&H1&+0:2]-[*:3].[*:4]-[N&H1&+0:5]-[*:6])"
    rxn2 = "[*:4]-[N&H1&+0:5](-[*:6])-C.[*:1]-[N&H0&+0:2](-[*:3])-C>>([*:1]-[N&H1&+0:2]-[*:3].[*:4]-[N&H1&+0:5]-[*:6])"
    rxn3 = "[N&H0&+0](-C)(-*)-*.[N&H1&+0](-C)(-*)-*>>([N&H1&+0](-*)-*.[N&H1&+0](-*)-*)"

    assert canon_reaction_smarts(rxn1, True) == canon_reaction_smarts(rxn2, True)
    assert (
        canon_reaction_smarts(rxn1)
        == canon_reaction_smarts(rxn2)
        == canon_reaction_smarts(rxn3)
    )

    rxn1 = "[N&H0&+0](-[O])(-*)-*.[P].[N&H0&+0](-[O;H1])(-*)-*>>([N&H1&+0](-*)-*.[N&H1&+0](-*)-*)"
    rxn2 = "[N&H0&+0](-[O])(-*)-*.[N&H0&+0](-[O;H1])(-*)-*.[P]>>([N&H1&+0](-*)-*.[N&H1&+0](-*)-*)"
    rxn3 = "[P].[N&H0&+0](-[O])(-*)-*.[N&H0&+0](-[O;H1])(-*)-*>>([N&H1&+0](-*)-*.[N&H1&+0](-*)-*)"

    assert canon_reaction_smarts(rxn1, True) == canon_reaction_smarts(rxn2, True)
    assert (
        canon_reaction_smarts(rxn1)
        == canon_reaction_smarts(rxn2)
        == canon_reaction_smarts(rxn3)
    )


def compare_products(reaction_template, reactants_in):
    canon_rxn1 = canon_reaction_smarts(reaction_template, True)
    rxn = AllChem.ReactionFromSmarts(reaction_template)
    rxn_canon = AllChem.ReactionFromSmarts(canon_rxn1)

    reactants = Chem.MolFromSmiles(reactants_in)
    p = rxn.RunReactants((reactants,))
    p2 = rxn_canon.RunReactants((reactants,))

    if len(p) != len(p2):
        return False

    p1s = []
    p1sms = []
    for l in p:
        for ll in l:
            Chem.SanitizeMol(ll)
            p1s.append(ll)
            sm_out = Chem.MolToSmiles(ll, isomericSmiles=False)
            sm_canon = Chem.CanonSmiles(sm_out)
            p1sms.append(sm_canon)

    p2s = []
    p2sms = []
    for l in p2:
        for ll in l:
            Chem.SanitizeMol(ll)
            p2s.append(ll)
            sm_out = Chem.MolToSmiles(ll, isomericSmiles=False)
            sm_canon = Chem.CanonSmiles(sm_out)
            p2sms.append(sm_canon)

    all_hit = True
    for ppp1, ppp2 in zip(sorted(p1sms), sorted(p2sms)):
        if ppp1 != ppp2:
            all_hit = False
            break

    if all_hit:
        return True
    else:
        return False


def test_run_reactions():
    rxn1 = "([*:1]-[N&H0&+0:2](-[*:3])-C.[*:4]-[N&H0&+0:5](-[*:6])-C)>>([*:1]-[N&H1&+0:2]-[*:3].[*:4]-[N&H1&+0:5]-[*:6])"
    reactant = "O=C(O)/C=C/C(=O)O.O=C(O)/C=C/C(=O)O.O=C(O)/C=C/C(=O)O.O=C1CCCN1CC#CCN1CCCC1.O=C1CCCN1CC#CCN1CCCC1"
    o = compare_products(rxn1, reactant)
    assert o

    rxn1 = "[#8]-[C&H1&+0:1](-[*:2])-[*:3]>>[*:2]-[C&H2&+0:1]-[*:3]"
    reactant = "C/C=C(\CCC(C)C1CCC2C3CC=C4CC(O)CCC4(C)C3CCC12C)C(C)C"
    o = compare_products(rxn1, reactant)
    assert o

    rxn1 = "[*:1]-[C&H2&+0:2]/[C&H0&+0:3](-[*:4])=[#7]/[N&H1&+0:5]-[*:6]:[c&H1&+0:7]:[*:8]>>[*:1]-[c&H0&+0:2]1:[c&H0&+0:3](-[*:4]):[n&H1&+0:5]:[*:6]:[c&H0&+0:7]:1:[*:8]"
    reactant = "O=[N+]([O-])c1ccc(NN=C2CCCCC2)c([N+](=O)[O-])c1"
    o = compare_products(rxn1, reactant)
    assert o


def test_against_random_permutations(in_smarts, n_perms=100):
    all_random = []
    for i in range(n_perms):
        sm = random_smarts(in_smarts)
        # print(sm)
        all_random.append(sm)

    all_random = set(all_random)

    canon_out = []
    for r in all_random:
        canon_out.append(canon_smarts(r))

    assert len(set(canon_out)) == 1


def test_random_permutations():
    path = os.path.dirname(os.path.abspath(__file__)) + "/testing_data/noncanon_efg_templates_20240108.xlsx"
    noncanon_templates = pd.read_excel(path)
    for t in noncanon_templates["noncanon_efg_templates"]:
        # print("Testing: ", t)
        test_against_random_permutations(t)


def test_multi_canon():
    path = os.path.dirname(os.path.abspath(__file__)) + "/testing_data/noncanon_efg_templates_20240108.xlsx"
    noncanon_templates = pd.read_excel(path)
    for t in noncanon_templates["noncanon_efg_templates"]:
        sm1 = canon_smarts(t)
        sm2 = canon_smarts(sm1)
        sm3 = canon_smarts(sm2)
        assert sm2 == sm3


def compare_substrate_datasets(template_smarts_dataset, substrate_smiles_dataset):
    for template_smarts, substrate_smiles in zip(
        template_smarts_dataset, substrate_smiles_dataset
    ):
        substrate_smiles.HasSubstructMatch(template_smarts)


def time_compare_substruct_match(
    template_smarts_dataset,
    substrate_smiles_dataset,
    embeddings=["askcos"],
    iters=10,
    v=False,
):
    noncanon_template_obj = [
        Chem.MolFromSmarts(template_smarts)
        for template_smarts in template_smarts_dataset
    ]
    mols = [
        Chem.MolFromSmiles(substrate_smiles)
        for substrate_smiles in substrate_smiles_dataset
    ]

    t1 = timeit.timeit(
        lambda: compare_substrate_datasets(noncanon_template_obj, mols),
        number=iters,
        timer=time.process_time,
    )

    ts = [t1]
    embeds = ["rdchiral"]
    for idx, emb in enumerate(embeddings):
        canon_template_obj = [
            Chem.MolFromSmarts(
                canon_smarts(template_smarts, mapping=True, embedding=emb)
            )
            for template_smarts in template_smarts_dataset
        ]

        t2 = timeit.timeit(
            lambda: compare_substrate_datasets(canon_template_obj, mols),
            number=iters,
            timer=time.process_time,
        )

        ts.append(t2)
        embeds.append("embed_" + str(idx + 2))

    if v:
        for t, e in zip(ts, embeds):
            print(e, t)
    return ts


def test_non_recursive_substruct_profile():
    path = os.path.dirname(os.path.abspath(__file__)) + "/testing_data/drugbank_non_matching_substruct_dataset_20240108.xlsx"
    noncanon_templates = pd.read_excel(path)

    times = time_compare_substruct_match(noncanon_templates["query_smarts"], 
                                         noncanon_templates["non_matching_substrate_smiles"], 
                                         embeddings=["askcos"], iters=100, v=False)

    print(times)
    assert times[1] < times[0]


def run_all_unit_tests():
    try:
        test_two_atom_smarts()
    except:
        raise Exception("Two atom smarts test failed")

    try:
        test_permutation_of_monosubstituted_benzene()
    except:
        raise Exception("Perumtation of monosubstituted benzene failed")

    try:
        test_symmetric_molecules()
    except:
        raise Exception("Symmetric molecules test failed")

    try:
        atom_invariant_permutations()
    except:
        raise Exception("Atom invariant permutations test failed")

    try:
        stereochemistry_permutations()
    except:
        raise Exception("Stereochemistry permutations test failed")

    try:
        test_recursive()
    except:
        raise Exception("Recursive test failed")
    
    try:
        test_random_permutations()
    except:
        raise Exception("Random permutations test failed")
    
    try:
        test_multi_canon()
    except:
        raise Exception("Multi canon test failed")

    try:
        test_permute_reactants()
    except:
        raise Exception("Permute reactants test failed")

    try:
        test_run_reactions()
    except:
        raise Exception("Run reactions test failed")
    
    try:
        test_non_recursive_substruct_profile()
    except:
        raise Exception("Non recursive substruct profile test failed")


def find_n_matches(smarts_library, target_library, n):
    out_data = {}

    for smol in smarts_library:
        sm = Chem.MolToSmarts(smol)
        if sm not in out_data:
            out_data[sm] = {
                "query_smarts": [],
                "matching_substrate_smiles": [],
                "non_matching_substrate_smiles": [],
                "random_substrate_smiles": [],
            }

        match_hit = False
        non_match_hit = False
        match_smiles = ""
        non_match_smiles = ""
        tot = 0
        for r in target_library:
            if r is None:
                continue

            if r.HasSubstructMatch(smol):
                match_hit = True
                match_smiles = Chem.MolToSmiles(r)
            else:
                non_match_hit = True
                non_match_smiles = Chem.MolToSmiles(r)

            if match_hit and non_match_hit:
                out_data[sm]["query_smarts"].append(sm)
                out_data[sm]["matching_substrate_smiles"].append(match_smiles)
                out_data[sm]["non_matching_substrate_smiles"].append(non_match_smiles)
                match_hit = False
                non_match_hit = False
                match_smiles = ""
                non_match_smiles = ""
                tot = tot + 1
            if tot == n:
                break

    return out_data


def validate_recursive(smarts_library, target_library, n, emb="askcos"):
    noncanon_template_obj = [
        Chem.MolFromSmarts(template_smarts) for template_smarts in smarts_library
    ]
    mols = [Chem.MolFromSmiles(substrate_smiles) for substrate_smiles in target_library]
    noncanon_output = find_n_matches(noncanon_template_obj, mols, n)

    canon_template_obj = [
        Chem.MolFromSmarts(canon_smarts(template_smarts, mapping=True, embedding=emb))
        for template_smarts in smarts_library
    ]

    canon_output = find_n_matches(canon_template_obj, mols, n)

    return noncanon_output, canon_output


def check_validation(noncanon_output, canon_output):
    for noncanon_hit, canon_hit in zip(
        noncanon_output,
        canon_output,
    ):
        
        if noncanon_output[noncanon_hit]['matching_substrate_smiles'] != canon_output[canon_hit]['matching_substrate_smiles']:
            print(noncanon_hit, canon_hit)
            return False
        
        if noncanon_output[noncanon_hit]['non_matching_substrate_smiles'] != canon_output[canon_hit]['non_matching_substrate_smiles']:
            print(noncanon_hit, canon_hit)
            return False

    return True


def run_validate_recursive_test():
    path = os.path.dirname(os.path.abspath(__file__)) + "/testing_data/noncanon_efg_templates_20240108.xlsx"
    noncanon_templates = pd.read_excel(path)
    path = os.path.dirname(os.path.abspath(__file__)) + "/testing_data/drugbank_all_structures_20231226.sdf"
    drugbank = Chem.SDMolSupplier(path)
    tdb = []
    for sm in drugbank:
        if sm is not None:
            tdb.append(Chem.MolToSmiles(sm))
        if len(tdb) > 1000: break

    nc, c = validate_recursive(noncanon_templates["noncanon_efg_templates"], tdb, 1)
    assert check_validation(nc, c)


def time_to_find_n_matches(
    smarts_library, target_library, n, embeddings=["askcos"], iters=10, v=False
):
    noncanon_template_obj = [
        Chem.MolFromSmarts(template_smarts) for template_smarts in smarts_library
    ]
    mols = [Chem.MolFromSmiles(substrate_smiles) for substrate_smiles in target_library]

    t1 = timeit.timeit(
        lambda: find_n_matches(noncanon_template_obj, mols, n),
        number=iters,
        timer=time.process_time,
    )

    ts = [t1]
    embeds = ["rdchiral"]
    for idx, emb in enumerate(embeddings):
        canon_template_obj = [
            Chem.MolFromSmarts(
                canon_smarts(template_smarts, mapping=True, embedding=emb)
            )
            for template_smarts in smarts_library
        ]

        t2 = timeit.timeit(
            lambda: find_n_matches(canon_template_obj, mols, n),
            number=iters,
            timer=time.process_time,
        )

        ts.append(t2)
        embeds.append("embed_" + str(idx + 2))

    if v:
        for t, e in zip(ts, embeds):
            print(e, t)
    return ts


def run_reactants_library(templates, substrates):
    for t, s in zip(templates, substrates):
        t.RunReactants((s,))


def time_compare_run_reactants(
    template_smarts, substrate_smiles, embeddings=["askcos"], iters=100000, v=False
):
    noncanon_template_objs = [
        AllChem.ReactionFromSmarts(template_smart) for template_smart in template_smarts
    ]

    substrate_objs = [
        Chem.MolFromSmiles(substrate_smile) for substrate_smile in substrate_smiles
    ]

    t1 = timeit.timeit(
        lambda: run_reactants_library(noncanon_template_objs, substrate_objs),
        number=iters,
        timer=time.process_time,
    )

    ts = [t1]
    embeds = ["rdchiral"]
    print("original done")
    for idx, emb in enumerate(embeddings):
        canon_template_obj = [
            AllChem.ReactionFromSmarts(
                canon_reaction_smarts(template_smart, mapping=True, embedding=emb)
            )
            for template_smart in template_smarts
        ]
        print(idx, "canon done")

        t2 = timeit.timeit(
            lambda: run_reactants_library(canon_template_obj, substrate_objs),
            number=iters,
            timer=time.process_time,
        )
        print(idx, "canon run done")

        ts.append(t2)
        embeds.append("embed_" + str(idx + 2))

    if v:
        for t, e in zip(ts, embeds):
            print(e, t)
    return ts


def compare_retrosim(template_smarts, template_smarts_obj, target_database_mols):
    for targ in target_database_mols:
        reaction_ran = []
        for idx, t in enumerate(template_smarts_obj):
            if template_smarts[idx] not in reaction_ran:
                p = t.RunReactants((targ,))
                reaction_ran.append(template_smarts[idx])


def time_compare_retrosyn(
    unsantized_template_smarts_dataset,
    target_database_smiles,
    embeddings=["askcos"],
    iters=10,
    v=False,
):
    unsantized_template_smarts_dataset_obj = [
        AllChem.ReactionFromSmarts(s) for s in unsantized_template_smarts_dataset
    ]
    target_database_mols = [Chem.MolFromSmiles(s) for s in target_database_smiles]

    t1 = timeit.timeit(
        lambda: compare_retrosim(
            unsantized_template_smarts_dataset,
            unsantized_template_smarts_dataset_obj,
            target_database_mols,
        ),
        number=iters,
        timer=time.process_time,
    )

    ts = [t1]
    embeds = ["rdchiral"]
    for idx, emb in enumerate(embeddings):
        canon_template_smarts_dataset_no_map = [
            canon_reaction_smarts(s, mapping=False, embedding=emb)
            for s in unsantized_template_smarts_dataset
        ]
        canon_template_smarts_dataset_map_obj = [
            AllChem.ReactionFromSmarts(
                canon_reaction_smarts(s, mapping=True, embedding=emb)
            )
            for s in unsantized_template_smarts_dataset
        ]

        t2 = timeit.timeit(
            lambda: compare_retrosim(
                canon_template_smarts_dataset_no_map,
                canon_template_smarts_dataset_map_obj,
                target_database_mols,
            ),
            number=iters,
            timer=time.process_time,
        )

        ts.append(t2)
        embeds.append("embed_" + str(idx + 2))

    if v:
        for t, e in zip(ts, embeds):
            print(e, t)
    return ts


def generate_1d_kdes(grid, data_in, bandwidth=0.01):
    # Scale the distributions
    data = np.array(data_in)

    kdes = []
    for idx, distribution in enumerate(data):
        kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(
            distribution.reshape(-1, 1)
        )

        grid = grid.reshape(-1, 1)

        kde_estimates = np.exp(
            kde.score_samples(grid)
        )  # score_samples returns log(density)

        kdes.append(kde_estimates)

    return kdes


def plot_kde(
    data,
    color_array,
    padding_percent=0.1,
    bandwidth=0.1,
    figsize=(3, 3),
    title="kde.png",
):
    pad = (np.max(data) - np.min(data)) * padding_percent

    x = np.linspace(np.min(data) - pad, np.max(data) + pad, 1000)

    kdes = generate_1d_kdes(x, data, bandwidth=bandwidth)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    for idx, kd in enumerate(kdes):
        ax.plot(x, kd, color=color_array[idx])
        ax.fill_between(x, kd, alpha=0.5, color=color_array[idx])
        ax.vlines(
            np.mean(data[idx]),
            0,
            np.max(kdes),
            color="black",
            linestyle="--",
            linewidth=1,
        )

    ax.set_yticks(ax.get_yticks())
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=6, fontfamily="arial")
    ax.set_xticks(ax.get_xticks()[1:-1])
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=6, fontfamily="arial")
    ax.set_ylim([0.01, np.max(kdes) + np.max(kdes) * 0.1])
    ax.set_xlabel(
        "time (cpu seconds/experiment)",
        fontsize=6,
        fontfamily="arial",
    )
    ax.set_ylabel("density", fontsize=6, fontfamily="arial")

    plt.savefig(title, dpi=300, bbox_inches="tight", pad_inches=0.01)
