from absl.testing import absltest
from rdcanon.main import canon_smarts, canon_reaction_smarts
from rdcanon.util import (
    compare_reaction_outputs,
    compare_products,
    run_against_library,
    compare_product_sets,
    run_random_permutations,
    time_compare_substruct_match,
)
from rdkit.Chem import AllChem
import pandas as pd
import os


class TestRegularSmarts(absltest.TestCase):
    def test_two_atom_smarts(self):
        s_test1 = "[C;H0;+0]-[C;H1;+0]"
        s_test2 = "[C;H1;+0]-[C;H0;+0]"
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test2))

        s_test1 = "[C;H0;+0]=[C;H1;+0]"
        s_test2 = "[C;H1;+0]=[C;H0;+0]"
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test2))

        s_test1 = "[C;H0;++]=[C;H1;+0]"
        s_test2 = "[C;H1;+0]=[C;H0;+2]"
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test2))

        s_test1 = "[C;H1;+0]-[C;H1;+0]"
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test1))

    def test_permutation_of_monosubstituted_benzene(self):
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
        for i in range(len(results)):
            for j in range(len(results)):
                self.assertEqual(results[i], results[j])

    def test_atom_invariant_permutations(self):
        s_test1 = "[Br;!Cl;H10;X10][c;H0]1[c;H0][c;H0][c;H0][c;H0][c;H0]1"
        s_test2 = "[!Cl;H10;X10;Br][c;H0]1[c;H0][c;H0][c;H0][c;H0][c;H0]1"
        s_test3 = "[H10;X10;Br;!Cl][c;H0]1[c;H0][c;H0][c;H0][c;H0][c;H0]1"
        s_test4 = "[X10;Br;!Cl;H10][c;H0]1[c;H0][c;H0][c;H0][c;H0][c;H0]1"

        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test2))
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test3))
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test4))

        s_test1 = "[Br;Cl,B,C]"
        s_test2 = "[Br;B,C,Cl]"
        s_test3 = "[Br;C,Cl,B]"
        s_test4 = "[Cl,B,C;Br]"
        s_test5 = "[B,C,Cl;Br]"
        s_test6 = "[C,Cl,B;Br]"

        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test2))
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test3))
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test4))
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test5))
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test6))

    def test_stereochemistry_permutations(self):
        s_test1 = "[C][C@][O]"
        s_test2 = "[C][C@@][O]"
        self.assertFalse(canon_smarts(s_test1) == canon_smarts(s_test2))

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
        self.assertTrue(len(set([canon_smarts(s) for s in isomers])) == 1)

        isomers = [
            "C(=C/O)\C-C(=C-[C@@](-Br)(-Cl)-[H1@&C](-B)(-P)-O)-C",
            "C(\O)=C/C-C(=C-[C@@](-[C@@&H1](-B)(-O)-P)(-Br)-Cl)-C",
            "Br-[C@](-Cl)(-[H1@@&C](-B)(-O)-P)-C=C(-C)-C\C=C\O",
            "C(-C)(-C\C=C\O)=C-[C@](-Br)(-[C@&H1](-B)(-P)-O)-Cl",
            "O\C=C\C-C(=C-[C@](-[H1@&C](-B)(-P)-O)(-Cl)-Br)-C",
            "C(=C-[C@](-Br)(-[H1@@&C](-B)(-O)-P)-Cl)(-C/C=C/O)-C",
            "B-[H1@&C](-[C@](-Br)(-Cl)-C=C(-C)-C/C=C/O)(-O)-P",
        ]
        self.assertTrue(len(set([canon_smarts(s) for s in isomers])) == 1)

    def test_symmetric_molecules(self):
        s_test1 = "[Cl][C][C][C][C][C][C][N][C][C][C][C][C][C][Br]"
        s_test2 = "[Br][C][C][C][C][C][C][N][C][C][C][C][C][C][Cl]"
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test2))

        s_test1 = "[Br][C][C][C][C][C][C][N][C][C][C][C][C][C]=[Br]"
        s_test2 = "[Br]=[C][C][C][C][C][C][N][C][C][C][C][C][C][Br]"
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test2))

        s_test1 = "[Cl][C][C][C][C][C][C][C][N][N][C][C][C][C][C][C][C][Br]"
        s_test2 = "[Br][C][C][C][C][C][C][C][N][N][C][C][C][C][C][C][C][Cl]"
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test2))

        s_test1 = "Clc1ccc(CCCCCCCNNCCCCCCCc2ccc(Br)cc2)cc1"
        s_test2 = "Brc1ccc(CCCCCCCNNCCCCCCCc2ccc(Cl)cc2)cc1"
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test2))

        s_test1 = "BrCCCCCCN(CCCCCCI)CCCCCCCl"
        s_test2 = "BrCCCCCCN(CCCCCCCl)CCCCCCI"
        s_test3 = "ClCCCCCCN(CCCCCCBr)CCCCCCI"
        s_test4 = "ClCCCCCCN(CCCCCCI)CCCCCCBr"
        s_test5 = "ICCCCCCN(CCCCCCBr)CCCCCCCl"
        s_test6 = "ICCCCCCN(CCCCCCCl)CCCCCCBr"

        expected_out = canon_smarts(s_test1)

        for s in [s_test2, s_test3, s_test4, s_test5, s_test6]:
            self.assertEqual(canon_smarts(s), expected_out)

        s_test1 = "[C]=,-[N]=[C]"
        s_test2 = "[C]=[N]=,-[C]"
        self.assertEqual(canon_smarts(s_test1), canon_smarts(s_test2))


class TestReactionSmarts(absltest.TestCase):
    def test_check_products_of_reactions(self):
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/testing_data/reaction_smarts_out.xlsx"
        )
        run_reactants_experiments = pd.read_excel(path)

        reactant_objs = [
            AllChem.MolFromSmiles(x)
            for x in run_reactants_experiments["matching_substrate"]
        ]
        extracted_smarts_objs = [
            AllChem.ReactionFromSmarts(x)
            for x in run_reactants_experiments["reaction_smarts"]
        ]
        canon_extracted_smarts_objs2 = [
            AllChem.ReactionFromSmarts(canon_reaction_smarts(x, True, "drugbank", True))
            for x in run_reactants_experiments["reaction_smarts"]
        ]

        correct, incorrect = compare_reaction_outputs(
            reactant_objs, extracted_smarts_objs, canon_extracted_smarts_objs2
        )
        self.assertTrue(incorrect == 0)

    def test_run_reactions(self):
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

    def test_permute_reactants(self):
        rxn1 = "[*:1]-[N&H0&+0:2](-[*:3])-C.[*:4]-[N&H0&+0:5](-[*:6])-C>>[*:1]-[N&H1&+0:2]-[*:3][C:7][N:8][*:4]-[N&H1&+0:5]-[*:6]"
        rxn2 = "[*:4]-[N&H0&+0:5](-[*:6])-C.[*:1]-[N&H0&+0:2](-[*:3])-C>>[*:1]-[N&H1&+0:2]-[*:3][C:7][N:8][*:4]-[N&H1&+0:5]-[*:6]"

        self.assertEqual(canon_reaction_smarts(rxn1), canon_reaction_smarts(rxn2))
        self.assertEqual(
            canon_reaction_smarts(rxn1, True), canon_reaction_smarts(rxn2, True)
        )
        self.assertEqual(
            canon_reaction_smarts(rxn1, True, "drugbank"),
            canon_reaction_smarts(rxn2, True, "drugbank"),
        )
        self.assertEqual(
            canon_reaction_smarts(rxn1, True, "drugbank", True),
            canon_reaction_smarts(rxn2, True, "drugbank", True),
        )

        rxn1 = "([*:1]-[N&H0&+0:2](-[*:3])-C.[*:4]-[N&H0&+0:5](-[*:6])-C)>>([*:1]-[N&H1&+0:2]-[*:3].[*:4]-[N&H1&+0:5]-[*:6])"
        rxn2 = "([*:4]-[N&H0&+0:5](-[*:6])-C.[*:1]-[N&H0&+0:2](-[*:3])-C)>>([*:1]-[N&H1&+0:2]-[*:3].[*:4]-[N&H1&+0:5]-[*:6])"
        rxn3 = "([N&H0&+0](-C)(-*)-*.[N&H0&+0](-C)(-*)-*)>>([N&H1&+0](-*)-*.[N&H1&+0](-*)-*)"

        self.assertEqual(
            canon_reaction_smarts(rxn1, True), canon_reaction_smarts(rxn2, True)
        )
        self.assertEqual(
            canon_reaction_smarts(rxn1, True, "drugbank", True),
            canon_reaction_smarts(rxn2, True, "drugbank", True),
        )
        self.assertEqual(canon_reaction_smarts(rxn1), canon_reaction_smarts(rxn3))

        rxn1 = "[*:1]-[N&H0&+0:2](-[*:3])-C.[*:4]-[N&H1&+0:5](-[*:6])-C>>([*:1]-[N&H1&+0:2]-[*:3].[*:4]-[N&H1&+0:5]-[*:6])"
        rxn2 = "[*:4]-[N&H1&+0:5](-[*:6])-C.[*:1]-[N&H0&+0:2](-[*:3])-C>>([*:1]-[N&H1&+0:2]-[*:3].[*:4]-[N&H1&+0:5]-[*:6])"
        rxn3 = (
            "[N&H0&+0](-C)(-*)-*.[N&H1&+0](-C)(-*)-*>>([N&H1&+0](-*)-*.[N&H1&+0](-*)-*)"
        )

        self.assertEqual(
            canon_reaction_smarts(rxn1, True), canon_reaction_smarts(rxn2, True)
        )
        self.assertEqual(canon_reaction_smarts(rxn1), canon_reaction_smarts(rxn3))

        rxn1 = "[N&H0&+0](-[O])(-*)-*.[P].[N&H0&+0](-[O;H1])(-*)-*>>([N&H1&+0](-*)-*.[N&H1&+0](-*)-*)"
        rxn2 = "[N&H0&+0](-[O])(-*)-*.[N&H0&+0](-[O;H1])(-*)-*.[P]>>([N&H1&+0](-*)-*.[N&H1&+0](-*)-*)"
        rxn3 = "[P].[N&H0&+0](-[O])(-*)-*.[N&H0&+0](-[O;H1])(-*)-*>>([N&H1&+0](-*)-*.[N&H1&+0](-*)-*)"

        self.assertEqual(
            canon_reaction_smarts(rxn1, True), canon_reaction_smarts(rxn2, True)
        )
        self.assertTrue(
            canon_reaction_smarts(rxn1)
            == canon_reaction_smarts(rxn2)
            == canon_reaction_smarts(rxn3)
        )

        rxn1 = "([C:1].[N:2])>>[C:1]#[N:2]"
        rxn2 = "([N:1].[C:2])>>[N:1]#[C:2]"

        self.assertEqual(
            canon_reaction_smarts(rxn1, True, "drugbank", True),
            canon_reaction_smarts(rxn2, True, "drugbank", True),
        )


class TestRecursive(absltest.TestCase):
    def test_recursive(self):
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
        s_test1 = canon_smarts("[$([#7R1]1-[#6R1]=[#7R1]-1)]")
        s_test2 = canon_smarts("[$([#7R1]1-[#7R1]=[#6R1]-1)]")
        assert canon_smarts(s_test1) == canon_smarts(s_test2)


    def test_multi_canon(self):
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/testing_data/noncanon_efg_templates_20240108.xlsx"
        )
        noncanon_templates = pd.read_excel(path)
        for t in noncanon_templates["noncanon_efg_templates"]:
            sm1 = canon_smarts(t)
            sm2 = canon_smarts(sm1)
            sm3 = canon_smarts(sm2)
            assert sm1 == sm2 == sm3

    def test_validate_recursive_against_database(self):
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/testing_data/noncanon_efg_templates_20240108.xlsx"
        )
        noncanon_templates = pd.read_excel(path)
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/testing_data/drugbank_smiles_1000.xlsx"
        )
        tdb = pd.read_excel(path)

        nc, c = run_against_library(
            noncanon_templates["noncanon_efg_templates"], tdb["smiles"], 1
        )
        assert compare_product_sets(nc, c)

    def test_random_permutations(self):
        # print(os.path.dirname(os.path.abspath(__file__)))
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/testing_data/noncanon_efg_templates_20240108.xlsx"
        )
        noncanon_templates = pd.read_excel(path)
        for t in noncanon_templates["noncanon_efg_templates"]:
            # print("Testing: ", t)
            assert run_random_permutations(t, n_perms=10)


class TestProfiling(absltest.TestCase):
    def test_non_recursive_substruct_profile(self):
        path = (
            os.path.dirname(os.path.abspath(__file__))
            + "/testing_data/drugbank_non_matching_substruct_dataset_20240108.xlsx"
        )
        noncanon_templates = pd.read_excel(path)

        times = time_compare_substruct_match(
            noncanon_templates["query_smarts"],
            noncanon_templates["non_matching_substrate_smiles"],
            embeddings=["drugbank"],
            iters=100,
            v=False,
        )

        print(times)
        assert times[1] < times[0]


if __name__ == "__main__":
    absltest.main()
