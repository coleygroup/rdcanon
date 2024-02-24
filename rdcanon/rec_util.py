from collections import deque
from functools import cmp_to_key
import rdkit
from rdkit import Chem
import re

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


class RecNode:
    def __init__(self, index, data):
        self.index = index
        self.data = data
        self.bonds = []  # Will hold Node instances
        self.bond_types = []
        self.bond_stereo = []
        self.bond_smarts = []
        self.serialized_score = []
        self.score_original = 0

    def add_bond(self, node, bond_type, bond_stereo, bond_smarts):
        if node not in self.bonds:
            self.bonds.append(node)
            self.bond_types.append(bond_type)
            self.bond_stereo.append(bond_stereo)
            self.bond_smarts.append(bond_smarts)

    def __repr__(self):
        return f"Node({self.index}, {self.data['smarts']})"


class RecGraph:
    def __init__(self, recursive_compare, v=False):
        self.nodes = []
        self.top_score = 0
        self.v = v
        self.recursive_compare = recursive_compare
        self.bond_indices_to_smarts = {}
        self.bond_indices_to_chiral = {}

    def custom_key2(self, item1t, item2t):
        item1 = item1t["path_scores"]
        item2 = item2t["path_scores"]

        if "unmapped_canon" in item1t:
            if item1t["unmapped_canon"] == item2t["unmapped_canon"]:
                sm1 = item1t["san_smarts"]
                sm2 = item2t["san_smarts"]
                return (sm1 > sm2) - (sm1 < sm2)

        return self.recursive_compare(item1, item2)

    def graph_from_smarts(self, mol, order_token_canon, embedding):
        for i,atom in enumerate(mol.GetAtoms()):
            node_data = {
                "symbol": atom.GetSymbol(),
                "atomic_num": atom.GetAtomicNum(),
                "smarts": atom.GetSmarts(),
                "stereo": atom.GetChiralTag(),
                "aromatic": atom.GetIsAromatic(),
                "hybridization": atom.GetHybridization(),
            }
            n = RecNode(atom.GetIdx(), node_data)


            sm, sc, _ = order_token_canon(
                n.data["smarts"], None, embedding
            )


            n.serialized_score = sc
            n.data["smarts"] = sm

            self.nodes.append(n)

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
                    bond.GetSmarts(),
                )
                self.nodes[end_idx].add_bond(
                    self.nodes[start_idx],
                    rdkit.Chem.rdchem.BondType.UNSPECIFIED,
                    bond.GetBondDir(),
                    bond.GetSmarts(),
                )
            else:
                self.nodes[start_idx].add_bond(
                    self.nodes[end_idx],
                    bond.GetBondType(),
                    bond.GetBondDir(),
                    bond.GetSmarts(),
                )
                self.nodes[end_idx].add_bond(
                    self.nodes[start_idx],
                    bond.GetBondType(),
                    bond.GetBondDir(),
                    bond.GetSmarts(),
                )

        for node in self.nodes:
            for i, bond in enumerate(node.bonds):
                self.bond_indices_to_smarts[(node.index, bond.index)] = (
                    node.bond_smarts[i]
                )
                self.bond_indices_to_smarts[(bond.index, node.index)] = (
                    node.bond_smarts[i]
                )
                self.bond_indices_to_chiral[(node.index, bond.index)] = (
                    node.bond_stereo[i]
                )
                self.bond_indices_to_chiral[(bond.index, node.index)] = (
                    node.bond_stereo[i]
                )

    def find_hamiltonian_paths_iterative(self, start_node, best_seen):
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
                    np = []
                    for rr in new_path:
                        np.append(rr[0].serialized_score)
                        if rr[1] == None:
                            bond_v = "None"
                        else:
                            bond_v = rr[1].name
                        np.append([bond_value_map[bond_v]])

                    if len(best_seen) == 0:
                        best_seen = np
                    else:
                        if len(np) > len(best_seen):
                            best_seen = np
                        elif len(np) < len(best_seen):
                            pass
                        else:
                            if (
                                min(
                                    [np, best_seen],
                                    key=cmp_to_key(self.recursive_compare),
                                )
                                == np
                            ):
                                best_seen = np
                            else:
                                continue

            for i, neighbor in enumerate(current_node.bonds):
                if neighbor.index not in visited:
                    nei = (neighbor, current_node.bond_types[i])

                    new_path = path + [nei]
                    np = []
                    for rr in new_path:
                        np.append(rr[0].serialized_score)
                        if rr[1] == None:
                            bond_v = "None"
                        else:
                            bond_v = rr[1].name
                        np.append([bond_value_map[bond_v]])

                    new_visited = []
                    for rr in visited:
                        new_visited.append(rr)
                    new_visited.append(neighbor.index)

                    if (
                        min([np, best_seen], key=cmp_to_key(self.recursive_compare))
                        == np
                    ):
                        stack.append((nei, new_path, new_visited, junction.copy()))

        return paths, best_seen

    def all_depth_first_search(self):
        poss_paths = []
        all_paths_scored = []
        path_idx = 0
        all_paths, best_seen = self.find_hamiltonian_paths_iterative(0, [])

        for r in all_paths:
            path_ar = []
            # print(r)
            for rr in r:
                # print(rr, rr[0])
                path_ar.append(rr[0].serialized_score)
                if rr[1] == None:
                    bond_v = "None"
                else:
                    bond_v = rr[1].name
                path_ar.append([bond_value_map[bond_v]])
            # print(path_ar)
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

        these_weights = sorted(all_paths_scored, key=cmp_to_key(self.custom_key2))

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

    def recreate_molecule(self):
        top_scores = self.all_depth_first_search()
        sms = []
        for top_score in top_scores:
            sms.append(
                (
                    self.regen_molecule(top_score["path"], False),
                    top_score["path_scores"],
                )
            )

        sms.sort(key=lambda x: x[0])
        self.top_score = sms[0][1]
        return sms[0]

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
        for node, bonde in dfs:
            for i, bond in enumerate(node.bonds):
                if (node.index, bond.index) in added or (
                    bond.index,
                    node.index,
                ) in added:
                    continue
                added.append((node.index, bond.index))

                # qbond = Chem.QueryBond()
                # qbond.SetBeginAtom(old_map_to_new_map[node.index])
                # qbond.SetEndAtom(old_map_to_new_map[bond.index])
                # qbond.SetQuery(node.bond_smarts[i])
                # mol.AddBond(qbond)
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

        node_order = []
        bond_order = []
        seen_bonds = []
        for i, r in enumerate(tmol.GetBonds()):
            this_node = tmol.GetAtomWithIdx(r.GetBeginAtomIdx()).GetAtomMapNum()
            next_node = tmol.GetAtomWithIdx(r.GetEndAtomIdx()).GetAtomMapNum()
            if (this_node, next_node) in seen_bonds or (
                next_node,
                this_node,
            ) in seen_bonds:
                continue
            seen_bonds.append((this_node, next_node))
            if str(r.GetBondDir()) == "ENDDOWNRIGHT":
                bond_order.append("\\")
            elif str(r.GetBondDir()) == "ENDUPRIGHT":
                bond_order.append("/")
            else:
                bond_order.append(self.bond_indices_to_smarts[(this_node, next_node)])

        for r in tmol.GetAtoms():
            if mapping:
                if r.GetAtomMapNum() in atom_map_number_to_true_atom_map_number:
                    new_map = atom_map_number_to_true_atom_map_number[r.GetAtomMapNum()]
                    r.SetAtomMapNum(new_map)
                else:
                    r.ClearProp("molAtomMapNumber")
            else:
                if "molAtomMapNumber" in r.GetPropsAsDict():
                    r.ClearProp("molAtomMapNumber")
                if r.GetAtomMapNum() == 0:
                    r.ClearProp("molAtomMapNumber")

            if str(r.GetChiralTag()) == "CHI_TETRAHEDRAL_CCW":

                sm = re.sub(r.GetSymbol(), r.GetSymbol() + "@", r.GetSmarts())
                if sm[0] == "[":
                    pass
                else:
                    sm = "[" + sm + "]"

                node_order.append(sm)
            elif str(r.GetChiralTag()) == "CHI_TETRAHEDRAL_CW":
                sm = re.sub(r.GetSymbol(), r.GetSymbol() + "@@", r.GetSmarts())

                if sm[0] == "[":
                    pass
                else:
                    sm = "[" + sm + "]"

                node_order.append(sm)

            else:
                node_order.append(r.GetSmarts())

        tmol_copy = Chem.MolFromSmarts(Chem.MolToSmarts(tmol))
        for r in tmol_copy.GetAtoms():
            if "molAtomMapNumber" in r.GetPropsAsDict():
                r.ClearProp("molAtomMapNumber")
            if r.GetAtomMapNum() == 0:
                r.ClearProp("molAtomMapNumber")

        self.unmapped_canon = Chem.MolToSmarts(tmol_copy)

        final_out = Chem.MolFragmentToSmiles(
            tmol,
            atomsToUse=range(len(node_order)),
            bondsToUse=range(len(bond_order)),
            atomSymbols=node_order,
            bondSymbols=bond_order,
            canonical=False,
        )

        return final_out
