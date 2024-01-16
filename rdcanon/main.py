from rdkit import Chem
from rdkit.Chem import AllChem
import re
from rdcanon.token_parser import order_token_canon, recursive_compare
import rdkit
from collections import deque
from rdcanon.askcos_prims import prims as prims1
import random
from functools import cmp_to_key


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
    "None": 2,
}


def custom_key2(item1t, item2t):
    item1 = item1t["path_scores"]
    item2 = item2t["path_scores"]

    return recursive_compare(item1, item2)


def custom_key3(item1t, item2t):
    item1 = item1t["score"]
    item2 = item2t["score"]

    return recursive_compare(item1, item2)


class Node:
    def __init__(self, index, data):
        self.index = index
        self.data = data
        self.bonds = []  # Will hold Node instances
        self.bond_types = []
        self.bond_stereo = []
        self.serialized_score = []
        self.score_original = 0

    def add_bond(self, node, bond_type, bond_stereo):
        if node not in self.bonds:
            self.bonds.append(node)
            self.bond_types.append(bond_type)
            self.bond_stereo.append(bond_stereo)

    def __repr__(self):
        return f"Node({self.index}, {self.data['smarts']})"


class Graph:
    def __init__(self, v=False):
        self.nodes = []
        self.mol = None
        self.top_score = 0
        self.v = v

    def graph_from_smarts(self, smarts, embedding):
        mol = Chem.MolFromSmarts(smarts)
        mol = Chem.MolFromSmarts(Chem.MolToSmarts(mol))

        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)

        self.mol = mol
        if not mol:
            raise ValueError("Invalid SMARTS provided")

        # Create nodes for each atom
        if self.v:
            print("input SMARTS:", smarts)
            print("rd-sanitized SMARTS:", Chem.MolToSmarts(mol))
            print()
            print("token embeddings")
        for atom in mol.GetAtoms():
            node_data = {
                "symbol": atom.GetSymbol(),
                "atomic_num": atom.GetAtomicNum(),
                "valence": atom.GetExplicitValence(),
                "smarts": atom.GetSmarts(),
                "stereo": atom.GetChiralTag(),
                "aromatic": atom.GetIsAromatic(),
                "hybridization": atom.GetHybridization(),
                # Add other desired atom properties here
            }
            n = Node(atom.GetIdx(), node_data)
            atom_map = re.findall(r":\d+]", n.data["smarts"])
            # print("incoming", n.data["smarts"])
            if len(atom_map) > 0:
                sm, sc, _ = order_token_canon(
                    re.sub(r":\d+]", "]", n.data["smarts"]),
                    atom_map[0][0:-1],
                    embedding,
                )
            else:
                sm, sc, _ = order_token_canon(
                    re.sub(r":\d+]", "]", n.data["smarts"]), None, embedding
                )

            if self.v:
                print(">", n.data["smarts"], sm, sc)

            single_score = sc
            while True:
                if type(single_score) == list or type(single_score) == tuple:
                    single_score = single_score[0]
                else:
                    break

            n.score_original = 1 / single_score
            n.serialized_score = sc
            n.data["smarts"] = sm
            self.nodes.append(n)
        if self.v:
            print()

        amol = Chem.MolFromSmarts("C:C")
        abond = amol.GetBondWithIdx(0)

        smol = Chem.MolFromSmarts("C-C")
        sbond = smol.GetBondWithIdx(0)

        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

            if bond.Match(abond) and bond.Match(sbond):
                self.nodes[start_idx].add_bond(
                    self.nodes[end_idx],
                    rdkit.Chem.rdchem.BondType.UNSPECIFIED,
                    bond.GetBondDir(),
                )
                self.nodes[end_idx].add_bond(
                    self.nodes[start_idx],
                    rdkit.Chem.rdchem.BondType.UNSPECIFIED,
                    bond.GetBondDir(),
                )
            else:
                self.nodes[start_idx].add_bond(
                    self.nodes[end_idx], bond.GetBondType(), bond.GetBondDir()
                )
                self.nodes[end_idx].add_bond(
                    self.nodes[start_idx], bond.GetBondType(), bond.GetBondDir()
                )

    def find_hamiltonian_paths_iterative(self, start_node):
        paths = []
        start_node = (self.nodes[start_node], None)
        pa = [start_node[0].index]
        stack = deque([(start_node, [start_node], pa, deque())])
        while stack:
            (curr_node, path, visited, junction) = stack.popleft()
            current_node, current_bond = curr_node
            if len(visited) == len(self.nodes):
                paths.append(path)
                continue

            all_neighbors_visited = True
            neighbors_not_visited = 0
            for r in current_node.bonds:
                if r.index not in visited:
                    all_neighbors_visited = False
                    neighbors_not_visited = neighbors_not_visited + 1

            if neighbors_not_visited > 1:
                junction.append((current_node, current_bond))

            if all_neighbors_visited:
                jnode = junction.pop()
                stack.append((jnode, path, visited, junction))

            for i, neighbor in enumerate(current_node.bonds):
                if neighbor.index not in visited:
                    nei = (neighbor, current_node.bond_types[i])
                    new_path = path + [nei]
                    new_visited = []
                    for rr in visited:
                        new_visited.append(rr)
                    new_visited.append(neighbor.index)
                    stack.append((nei, new_path, new_visited, junction.copy()))

        return paths

    def all_depth_first_search(self):
        if self.v:
            print("enumerated paths")

        node_data = [x.serialized_score for x in self.nodes]

        n = min(node_data, key=cmp_to_key(recursive_compare))
        top_nodes = []
        for i, nd in enumerate(node_data):
            if nd == n:
                top_nodes.append(i)

        poss_paths = []
        all_paths_scored = []
        path_idx = 0
        for idx, h in enumerate(self.nodes):
            if idx not in top_nodes:
                continue
            all_paths = self.find_hamiltonian_paths_iterative(h.index)
            for r in all_paths:
                path_ar = []
                for rr in r:
                    path_ar.append(rr[0].serialized_score)
                    if rr[1] == None:
                        bond_v = "None"
                    else:
                        bond_v = rr[1].name
                    path_ar.append([bond_value_map[bond_v]])
                all_paths_scored.append(
                    {
                        "path_scores": path_ar,
                        "path": r,
                    }
                )
                if self.v:
                    print("> path", path_idx)
                    for rr in r:
                        print(">>", rr[0], rr[0].serialized_score, rr[1])
                    print(">>", path_ar)

                poss_paths.append(r)
                path_idx = path_idx + 1

        if self.v:
            print()
            print("unsorted paths")
            for kk in all_paths_scored:
                print(">", kk)

        these_weights = sorted(all_paths_scored, key=cmp_to_key(custom_key2))

        if self.v:
            print()
            print("sorted paths")
            for kk in these_weights:
                print(">", kk)

        top_tied = []
        top_score = these_weights[0]["path_scores"]
        for i, p in enumerate(these_weights):
            if p["path_scores"] == top_score:
                top_tied.append(p)
            else:
                break

        return top_tied

    def can_transform(self, set1, set2):
        """
        Check if one set of indices can be transformed into another set by rotating three of the indices
        around a stationary one.
        """
        if len(set1) != len(set2) or len(set1) > 4:
            return False

        lst = set1[1:]
        n = len(lst)
        arrs = []
        for i in range(n):
            # Rotate list by i places
            rotated_list = lst[-i:] + lst[:-i]
            this_ar = set1[0]
            rotated_list.insert(0, this_ar)
            arrs.append(rotated_list)

        new_arrs = []
        for arr in arrs:
            new_arrs.append(arr)
            rev = list(reversed(arr))
            lst = rev[1:]
            n = len(lst)
            for i in range(n):
                rotated_list = lst[-i:] + lst[:-i]
                this_ar = rev[0]
                rotated_list.insert(0, this_ar)
                new_arrs.append(rotated_list)

        for arr in new_arrs:
            if arr == set2:
                return True
        return False

    def recreate_molecule(self, mapping):
        top_scores = self.all_depth_first_search()
        sms = []
        for top_score in top_scores:
            sms.append(
                (
                    self.regen_molecule(top_score["path"], False),
                    self.regen_molecule(top_score["path"], mapping),
                    top_score["path_scores"],
                )
            )

        sms.sort(key=lambda x: x[0])
        self.top_score = sms[0][2]
        return sms[0][1]

    def regen_molecule(self, dfs, mapping):
        mol = Chem.RWMol()
        old_map_to_new_map = {}
        i = 0
        idxes_out = []
        new_map_to_old_map = {}
        atom_map_number_to_true_atom_map_number = {}
        for node, bond in dfs:
            sm = node.data["smarts"]
            atom_map = re.findall(r":\d+", sm)
            if not mapping:
                sm = re.sub(r":\d+]", "]", sm)
            smarts = sm
            mol2 = Chem.MolFromSmarts(smarts)
            atom = mol2.GetAtomWithIdx(0)
            atom.SetChiralTag(node.data["stereo"])
            atom.SetAtomMapNum(node.index)
            if len(atom_map) > 0 and mapping:
                atom_map_number_to_true_atom_map_number[node.index] = int(
                    atom_map[0][1:]
                )
            mol.AddAtom(atom)
            old_map_to_new_map[node.index] = i
            new_map_to_old_map[i] = node.index
            idxes_out.append(node.index)
            i = i + 1
        added = []
        for node, bond in dfs:
            for i, bond in enumerate(node.bonds):
                if (node.index, bond.index) in added or (
                    bond.index,
                    node.index,
                ) in added:
                    continue
                added.append((node.index, bond.index))
                mol.AddBond(
                    old_map_to_new_map[node.index],
                    old_map_to_new_map[bond.index],
                    node.bond_types[i],
                )
                bond = list(mol.GetBonds())[-1]
                bond.SetBondDir(node.bond_stereo[i])

        og = Chem.MolToSmarts(mol)
        tmol = Chem.MolFromSmarts(og)

        # print("original:", og)

        cw_ord = [0, 1, 2]
        ccw_ord = [0, 2, 1]

        for i, atm in enumerate(tmol.GetAtoms()):
            if (
                atm.GetChiralTag() == rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
                or atm.GetChiralTag()
                == rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
            ):
                old_bonds_indices = [
                    nn.index for nn in self.nodes[atm.GetAtomMapNum()].bonds
                ]
                new_bond_indices = []
                for nn in atm.GetNeighbors():
                    if "molAtomMapNumber" in nn.GetPropsAsDict():
                        new_bond_indices.append(int(nn.GetProp("molAtomMapNumber")))
                    else:
                        new_bond_indices.append(0)

                if len(old_bonds_indices) == 3:
                    old_bonds_indices.insert(1, -1)
                    new_bond_indices.insert(1, -1)
                # print(atm.GetAtomMapNum())
                # print("old bonds:", old_bonds_indices)
                # print("new bonds:", new_bond_indices)
                if (
                    self.nodes[atm.GetAtomMapNum()].data["stereo"]
                    == rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
                ):
                    if (
                        atm.GetChiralTag()
                        == rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
                    ):
                        # print("old: CW (@@)", "new: CW (@@)")
                        new_atms = []
                        old_atms = []
                        for f_idx in range(len(new_bond_indices)):
                            if f_idx == 0:
                                continue
                            new_atm = new_bond_indices[f_idx]
                            old_atm = old_bonds_indices[f_idx]
                            new_atms.append(new_atm)
                            old_atms.append(old_atm)
                        new_ordered = [new_bond_indices[0]]
                        old_ordered = [old_bonds_indices[0]]
                        for j in range(len(new_atms)):
                            new_ordered.append(new_atms[cw_ord[j]])
                            old_ordered.append(old_atms[cw_ord[j]])
                        # print("old ordered:", old_ordered)
                        # print("new ordered:", new_ordered)
                        if self.can_transform(old_ordered, new_ordered):
                            # print("permutable -> keep new as CW")
                            pass
                        else:
                            # print("not permutable -> set new to CCW")
                            atm.SetChiralTag(
                                rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
                            )
                    else:
                        # print("old: CW (@@)", "new: CCW (@)")
                        new_atms = []
                        old_atms = []
                        for f_idx in range(len(new_bond_indices)):
                            if f_idx == 0:
                                continue
                            new_atm = new_bond_indices[f_idx]
                            old_atm = old_bonds_indices[f_idx]
                            new_atms.append(new_atm)
                            old_atms.append(old_atm)
                        new_ordered = [new_bond_indices[0]]
                        old_ordered = [old_bonds_indices[0]]
                        for j in range(len(new_atms)):
                            new_ordered.append(new_atms[ccw_ord[j]])
                            old_ordered.append(old_atms[cw_ord[j]])
                        # print("old ordered:", old_ordered)
                        # print("new ordered:", new_ordered)
                        if self.can_transform(old_ordered, new_ordered):
                            # print("permutable -> keep new as CCW")
                            pass
                        else:
                            # print("not permutable -> set new to CW")
                            atm.SetChiralTag(
                                rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
                            )
                else:
                    if (
                        atm.GetChiralTag()
                        == rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
                    ):
                        # print("old: CCW (@)", "new: CW (@@)")
                        new_atms = []
                        old_atms = []
                        for f_idx in range(len(new_bond_indices)):
                            if f_idx == 0:
                                continue
                            new_atm = new_bond_indices[f_idx]
                            old_atm = old_bonds_indices[f_idx]
                            new_atms.append(new_atm)
                            old_atms.append(old_atm)
                        new_ordered = [new_bond_indices[0]]
                        old_ordered = [old_bonds_indices[0]]
                        for j in range(len(new_atms)):
                            new_ordered.append(new_atms[cw_ord[j]])
                            old_ordered.append(old_atms[ccw_ord[j]])
                        # print("old ordered:", old_ordered)
                        # print("new ordered:", new_ordered)
                        if self.can_transform(old_ordered, new_ordered):
                            # print("permutable -> keep new as CW")
                            pass
                        else:
                            # print("not permutable -> set new to CCW")
                            atm.SetChiralTag(
                                rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW
                            )
                    else:
                        # print("old: CCW (@)", "new: CCW (@)")
                        new_atms = []
                        old_atms = []
                        for f_idx in range(len(new_bond_indices)):
                            if f_idx == 0:
                                continue
                            new_atm = new_bond_indices[f_idx]
                            old_atm = old_bonds_indices[f_idx]
                            new_atms.append(new_atm)
                            old_atms.append(old_atm)
                        new_ordered = [new_bond_indices[0]]
                        old_ordered = [old_bonds_indices[0]]
                        for j in range(len(new_atms)):
                            new_ordered.append(new_atms[ccw_ord[j]])
                            old_ordered.append(old_atms[ccw_ord[j]])
                        # print("old ordered:", old_ordered)
                        # print("new ordered:", new_ordered)
                        if self.can_transform(old_ordered, new_ordered):
                            pass
                            # print("permutable -> keep new as CCW")
                        else:
                            # print("not permutable -> set new to CW")
                            atm.SetChiralTag(
                                rdkit.Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW
                            )

        for r in tmol.GetAtoms():
            if mapping:
                if r.GetAtomMapNum() in atom_map_number_to_true_atom_map_number:
                    r.SetAtomMapNum(
                        atom_map_number_to_true_atom_map_number[r.GetAtomMapNum()]
                    )
                else:
                    r.ClearProp("molAtomMapNumber")
            else:
                if "molAtomMapNumber" in r.GetPropsAsDict():
                    r.ClearProp("molAtomMapNumber")
                if r.GetAtomMapNum() == 0:
                    r.ClearProp("molAtomMapNumber")

        return Chem.MolToSmarts(tmol)

    def __repr__(self):
        return f"Graph({self.nodes})"


def get_sanitized_reaction_smarts(sm2, mapping, embedding):
    k = AllChem.ReactionFromSmarts(sm2)
    reactants = []
    agents = []
    products = []
    for r in k.GetReactants():
        r_sm = Chem.MolToSmarts(r)

        if "." in r_sm:
            smss = r_sm.split(".")
        else:
            smss = [r_sm]
        tss = []
        sans = []
        for sm in smss:
            san_sm, ts = canon_smarts(sm, mapping, embedding, return_score=True)
            tss.append(ts)
            sans.append(san_sm)

        san_sm_out = ".".join(sans)
        if "." in r_sm:
            san_sm_out = "(" + san_sm_out + ")"

        # print(r_sm, san_sm_out, ts)
        reactants.append(
            {
                "path_scores": tss,
                "original_smarts": r_sm,
                "san_smarts": san_sm_out,
            }
        )

    for r in k.GetAgents():
        r_sm = Chem.MolToSmarts(r)
        if "." in r_sm:
            smss = r_sm.split(".")
        else:
            smss = [r_sm]
        tss = []
        sans = []
        for sm in smss:
            san_sm, ts = canon_smarts(sm, mapping, embedding, return_score=True)
            tss.append(ts)
            sans.append(san_sm)

        san_sm_out = ".".join(sans)
        if "." in r_sm:
            san_sm_out = "(" + san_sm_out + ")"

        agents.append(
            {
                "path_scores": tss,
                "original_smarts": r_sm,
                "san_smarts": san_sm_out,
            }
        )

    for r in k.GetProducts():
        r_sm = Chem.MolToSmarts(r)
        if "." in r_sm:
            smss = r_sm.split(".")
        else:
            smss = [r_sm]
        tss = []
        sans = []
        for sm in smss:
            san_sm, ts = canon_smarts(sm, mapping, embedding, return_score=True)
            tss.append(ts)
            sans.append(san_sm)

        san_sm_out = ".".join(sans)
        if "." in r_sm:
            san_sm_out = "(" + san_sm_out + ")"

        products.append(
            {
                "path_scores": tss,
                "original_smarts": r_sm,
                "san_smarts": san_sm_out,
            }
        )

    reactants_sort = sorted(reactants, key=cmp_to_key(custom_key2))

    san_smarts_out = ".".join([r["san_smarts"] for r in reactants_sort])

    if len(agents) > 0:
        san_smarts_out = san_smarts_out + ">"
        agents_sort = sorted(agents, key=cmp_to_key(custom_key2))
        san_smarts_out = san_smarts_out + ".".join(
            [r["san_smarts"] for r in agents_sort]
        )

    san_smarts_out = san_smarts_out + ">>"
    products_sort = sorted(products, key=cmp_to_key(custom_key2))
    san_smarts_out = san_smarts_out + ".".join([r["san_smarts"] for r in products_sort])

    # print(sm2)
    # print(san_smarts_out)

    return san_smarts_out


def random_smarts(smarts="[Cl][C][C][C][N][C][C][C][Br]", mapping=False):
    """
    Generate a random molecule based on the given SMARTS pattern.

    Args:
        smarts (str): The SMARTS pattern representing the molecule structure. Default is "[Cl][C][C][C][N][C][C][C][Br]".
        mapping (bool): Whether to generate a mapping of atom indices between the original and recreated molecule. Default is False.

    Returns:
        str: The recreated SMARTS pattern.

    """
    g = Graph()

    prims = {}
    for k in prims1:
        prims[k] = random.random()

    g.graph_from_smarts(smarts, prims)

    return g.recreate_molecule(mapping)


def canon_smarts(
    smarts, mapping=False, embedding="drugbank", return_score=False, v=False
):
    """
    Canonicalizes a SMARTS pattern.

    Args:
        smarts (str): The input SMARTS pattern to be canonicalized.
        mapping (bool, optional): Whether to return the atom mapping. Defaults to False.
        embedding (str, optional): The query primitive frequency dictionary to use. Defaults to "drugbank".
        return_score (bool, optional): Whether to return the top score. Defaults to False.
        v (bool, optional): Whether to enable verbose mode. Defaults to False.

    Returns:
        str or tuple: The canonicalized SMARTS pattern. If `return_score` is True, a tuple containing the canonicalized SMARTS pattern and the top score is returned.
    """
    g = Graph(v)
    g.graph_from_smarts(smarts, embedding)
    out = g.recreate_molecule(mapping)
    if return_score:
        return out, g.top_score
    return out


def debug(smarts, mapping=False, embedding="drugbank", return_score=False):
    canon_smarts(smarts, mapping, embedding, return_score, True)


def canon_reaction_smarts(smarts, mapping=False, embedding="drugbank"):
    """
    Canonicalizes a reaction SMARTS string.

    Args:
        smarts (str): The reaction SMARTS string to be canonicalized.
        mapping (bool, optional): Whether to include atom mapping in the canonicalization. Defaults to False.
        embedding (str, optional): The embedding to use for the canonicalization. Defaults to "drugbank".

    Returns:
        str: The canonicalized reaction SMARTS string.
    """
    san_smarts_out = get_sanitized_reaction_smarts(smarts, mapping, embedding)
    return san_smarts_out
