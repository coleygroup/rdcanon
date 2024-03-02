from collections import deque
from functools import cmp_to_key
import rdkit
from rdkit import Chem
import re
from rdkit.Chem.rdchem import BondType, BondDir, BondStereo

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
        self.bond_indices_to_stereo = {}
        self.bond_indices_to_relative_stereo = {}

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
        mol = Chem.MolFromSmarts(mol)
        for i, atom in enumerate(mol.GetAtoms()):
            node_data = {
                "smarts": atom.GetSmarts(),
                "stereo": atom.GetChiralTag(),
            }
            n = RecNode(atom.GetIdx(), node_data)

            sm, sc, _ = order_token_canon(n.data["smarts"], None, embedding)

            n.serialized_score = sc
            n.data["smarts"] = sm

            self.nodes.append(n)

        rdkit.Chem.rdmolops.FastFindRings(mol)
        rdkit.Chem.rdmolops.FindPotentialStereoBonds(mol)

        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()

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

            self.bond_indices_to_stereo[(start_idx, end_idx)] = bond.GetStereo()
            self.bond_indices_to_stereo[(end_idx, start_idx)] = bond.GetStereo()

            bond_a = None
            bond_b = None

            if bond.GetStereo() != Chem.rdchem.BondStereo.STEREONONE:
                atom = mol.GetAtomWithIdx(start_idx)
                neighbors = atom.GetNeighbors()

                for neighbor in neighbors:
                    if (
                        mol.GetBondBetweenAtoms(
                            start_idx, neighbor.GetIdx()
                        ).GetBondDir()
                        == Chem.rdchem.BondDir.ENDUPRIGHT
                        or mol.GetBondBetweenAtoms(
                            start_idx, neighbor.GetIdx()
                        ).GetBondDir()
                        == Chem.rdchem.BondDir.ENDDOWNRIGHT
                    ):
                        bond_a = neighbor.GetIdx()
                        break

                atom = mol.GetAtomWithIdx(end_idx)
                neighbors = atom.GetNeighbors()

                for neighbor in neighbors:
                    if (
                        mol.GetBondBetweenAtoms(end_idx, neighbor.GetIdx()).GetBondDir()
                        == Chem.rdchem.BondDir.ENDUPRIGHT
                        or mol.GetBondBetweenAtoms(
                            end_idx, neighbor.GetIdx()
                        ).GetBondDir()
                        == Chem.rdchem.BondDir.ENDDOWNRIGHT
                    ):
                        bond_b = neighbor.GetIdx()
                        break

            if bond_a is not None and bond_b is not None:
                self.bond_indices_to_relative_stereo[(bond_a, bond_b)] = (
                    bond.GetStereo()
                )
                self.bond_indices_to_relative_stereo[(bond_b, bond_a)] = (
                    bond.GetStereo()
                )

            self.bond_indices_to_smarts[(start_idx, end_idx)] = bond.GetSmarts()
            self.bond_indices_to_smarts[(end_idx, start_idx)] = bond.GetSmarts()

    def replace_at_index(self, original, new_text, start, length):
        end = start + length
        return original[:start] + new_text + original[end:]

    def insert_at_index(self, original, new_text, start):
        return original[:start] + new_text + original[start:]

    def find_hamiltonian_paths_iterative_sm(self, start_node, best_seen):
        paths = []
        smiles_out = []
        node_maps_out = []
        bond_maps_out = []

        start_node = (self.nodes[start_node], None, None)

        sm_so_far1 = start_node[0].data["smarts"]

        pa = [start_node[0].index]
        node_to_sm_idx = {start_node[0].index: len(sm_so_far1)}
        bond_to_sm_idx = {}
        branch_level = 0
        stack = deque(
            [
                (
                    start_node,
                    [start_node],
                    pa,
                    deque(),
                    sm_so_far1,
                    None,
                    node_to_sm_idx,
                    1,
                    bond_to_sm_idx,
                    branch_level,
                )
            ]
        )

        while stack:
            (
                curr_node,
                path,
                visited,
                junction,
                sm_so_far,
                parent_index,
                this_node_to_sm_idx,
                ring_num,
                this_bond_to_sm_idx,
                this_branch_level,
            ) = stack.popleft()
            current_node, current_bond, curr_bond_smarts = curr_node
            if len(visited) == len(self.nodes):
                for r in current_node.bonds:
                    if r.index != parent_index and parent_index != -1:
                        bond_sm = self.bond_indices_to_smarts[
                            (current_node.index, r.index)
                        ]
                        sm_so_far = sm_so_far + bond_sm
                        this_bond_to_sm_idx[(current_node.index, r.index)] = len(
                            sm_so_far
                        )
                        sm_so_far = sm_so_far + str(ring_num)
                        sm_so_far = self.insert_at_index(
                            sm_so_far, str(ring_num), this_node_to_sm_idx[r.index]
                        )

                        for nn in this_node_to_sm_idx:
                            if this_node_to_sm_idx[nn] > this_node_to_sm_idx[r.index]:
                                this_node_to_sm_idx[nn] = this_node_to_sm_idx[nn] + len(
                                    str(ring_num)
                                )

                        for bb in this_bond_to_sm_idx:
                            if this_bond_to_sm_idx[bb] > this_node_to_sm_idx[r.index]:
                                this_bond_to_sm_idx[bb] = this_bond_to_sm_idx[bb] + len(
                                    str(ring_num)
                                )

                        ring_num = ring_num + 1

                if this_branch_level > 0:
                    for i in range(this_branch_level):
                        sm_so_far = sm_so_far + ")"

                paths.append(path)
                smiles_out.append(sm_so_far)
                node_maps_out.append(this_node_to_sm_idx)
                bond_maps_out.append(this_bond_to_sm_idx)
                continue

            all_neighbors_visited = True
            neighbors_not_visited = 0
            for r in current_node.bonds:
                if r.index not in visited:
                    all_neighbors_visited = False
                    neighbors_not_visited = neighbors_not_visited + 1
                else:
                    if r.index != parent_index and parent_index != -1:

                        bond_sm = self.bond_indices_to_smarts[
                            (current_node.index, r.index)
                        ]

                        sm_so_far = sm_so_far + bond_sm
                        this_bond_to_sm_idx[(current_node.index, r.index)] = len(
                            sm_so_far
                        )
                        sm_so_far = sm_so_far + str(ring_num)
                        sm_so_far = self.insert_at_index(
                            sm_so_far, str(ring_num), this_node_to_sm_idx[r.index]
                        )

                        for nn in this_node_to_sm_idx:
                            if this_node_to_sm_idx[nn] > this_node_to_sm_idx[r.index]:
                                this_node_to_sm_idx[nn] = this_node_to_sm_idx[nn] + len(
                                    str(ring_num)
                                )

                        for bb in this_bond_to_sm_idx:
                            if this_bond_to_sm_idx[bb] > this_node_to_sm_idx[r.index]:
                                this_bond_to_sm_idx[bb] = this_bond_to_sm_idx[bb] + len(
                                    str(ring_num)
                                )

                        ring_num = ring_num + 1

            if neighbors_not_visited > 1:
                this_branch_level = this_branch_level + 1
                sm_so_far = sm_so_far + "("
                junction.append((current_node, current_bond, curr_bond_smarts))

            if all_neighbors_visited:
                jnode = junction.pop()
                sm_so_far = sm_so_far + ")"
                this_branch_level = this_branch_level - 1
                stack.append(
                    (
                        jnode,
                        path,
                        visited,
                        junction,
                        sm_so_far,
                        -1,
                        this_node_to_sm_idx,
                        ring_num,
                        this_bond_to_sm_idx,
                        this_branch_level,
                    )
                )

            for i, neighbor in enumerate(current_node.bonds):
                if neighbor.index not in visited:
                    nei = (
                        neighbor,
                        current_node.bond_types[i],
                        current_node.bond_smarts[i],
                    )
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
                    nei = (
                        neighbor,
                        current_node.bond_types[i],
                        current_node.bond_smarts[i],
                    )

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
                        tsm_so_far = sm_so_far + current_node.bond_smarts[i]
                        new_this_bond_to_sm_idx = this_bond_to_sm_idx.copy()
                        new_this_bond_to_sm_idx[
                            (current_node.index, neighbor.index)
                        ] = len(tsm_so_far)
                        tsm_so_far = tsm_so_far + neighbor.data["smarts"]

                        new_this_node_to_sm_idx = this_node_to_sm_idx.copy()

                        new_this_node_to_sm_idx[neighbor.index] = len(tsm_so_far)

                        stack.append(
                            (
                                nei,
                                new_path,
                                new_visited,
                                junction.copy(),
                                tsm_so_far,
                                current_node.index,
                                new_this_node_to_sm_idx,
                                ring_num,
                                new_this_bond_to_sm_idx,
                                this_branch_level,
                            )
                        )

        return paths, best_seen, smiles_out, node_maps_out, bond_maps_out

    def all_depth_first_search(self):
        poss_paths = []
        all_paths_scored = []
        path_idx = 0
        all_paths, best_seen, sm_o, node_map, bond_map = (
            self.find_hamiltonian_paths_iterative_sm(0, [])
        )

        for i, r in enumerate(all_paths):
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
                    "smarts": sm_o[i],
                    "node_map": node_map[i],
                    "bond_map": bond_map[i],
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
            unmapped, mapped = self.regen_molecule(
                top_score["path"],
                top_score["smarts"],
                top_score["node_map"],
                top_score["bond_map"],
            )

            sms.append(
                (
                    unmapped,
                    mapped,
                    top_score["path_scores"],
                )
            )

        sms.sort(key=lambda x: x[0])
        self.top_score = sms[0][2]
        return sms[0][0], sms[0][2]

    def regen_molecule(self, dfs, smarts_in, node_map, bond_map):
        old_map_to_new_map = {}
        i = 0
        idxes_out = []
        new_map_to_old_map = {}
        atom_map_number_to_true_atom_map_number = {}

        tmol = Chem.MolFromSmarts(smarts_in)
        for atom, (node, _, _) in zip(tmol.GetAtoms(), dfs):
            sm = node.data["smarts"]
            atom_map = re.findall(r":\d+", sm)
            atom.SetChiralTag(node.data["stereo"])
            atom.SetAtomMapNum(node.index)
            if len(atom_map) > 0:
                atom_map_number_to_true_atom_map_number[node.index] = int(
                    atom_map[0][1:]
                )

            old_map_to_new_map[node.index] = i
            new_map_to_old_map[i] = node.index
            idxes_out.append(node.index)
            i = i + 1

        bond_order = []
        bonds_set_equal = []
        bonds_set_trans = []
        for bond in tmol.GetBonds():
            bond_order.append(
                self.bond_indices_to_smarts[
                    (
                        tmol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum(),
                        tmol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum(),
                    )
                ]
            )
            if bond.GetBondType() == BondType.DOUBLE:
                start_atom = tmol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                end_atom = tmol.GetAtomWithIdx(bond.GetEndAtomIdx())

                start_rs = []
                end_rs = []
                for nene in start_atom.GetNeighbors():
                    if nene.GetIdx() == end_atom.GetIdx():
                        continue
                    start_rs.append(nene.GetAtomMapNum())
                for nene in end_atom.GetNeighbors():
                    if nene.GetIdx() == start_atom.GetIdx():
                        continue
                    end_rs.append(nene.GetAtomMapNum())

                all_r_combos = []
                for i in start_rs:
                    for j in end_rs:
                        all_r_combos.append((i, j))

                for r in all_r_combos:
                    if r in self.bond_indices_to_relative_stereo:
                        if (
                            self.bond_indices_to_relative_stereo[r]
                            == BondStereo.STEREOCIS
                        ):
                            if r[0] in start_rs:
                                bond_a = tmol.GetBondBetweenAtoms(
                                    start_atom.GetIdx(), old_map_to_new_map[r[0]]
                                ).GetIdx()
                                bond_b = tmol.GetBondBetweenAtoms(
                                    end_atom.GetIdx(), old_map_to_new_map[r[1]]
                                ).GetIdx()
                            else:
                                bond_a = tmol.GetBondBetweenAtoms(
                                    start_atom.GetIdx(), old_map_to_new_map[r[1]]
                                ).GetIdx()
                                bond_b = tmol.GetBondBetweenAtoms(
                                    end_atom.GetIdx(), old_map_to_new_map[r[0]]
                                ).GetIdx()
                            bonds_set_equal.append((bond_a, bond_b))
                        elif (
                            self.bond_indices_to_relative_stereo[r]
                            == BondStereo.STEREOTRANS
                        ):
                            if r[0] in start_rs:
                                bond_a = tmol.GetBondBetweenAtoms(
                                    start_atom.GetIdx(), old_map_to_new_map[r[0]]
                                ).GetIdx()
                                bond_b = tmol.GetBondBetweenAtoms(
                                    end_atom.GetIdx(), old_map_to_new_map[r[1]]
                                ).GetIdx()
                            else:
                                bond_a = tmol.GetBondBetweenAtoms(
                                    start_atom.GetIdx(), old_map_to_new_map[r[1]]
                                ).GetIdx()
                                bond_b = tmol.GetBondBetweenAtoms(
                                    end_atom.GetIdx(), old_map_to_new_map[r[0]]
                                ).GetIdx()
                            bonds_set_trans.append((bond_a, bond_b))

        for bond in bonds_set_equal:
            bond = sorted(bond)
            bond_order[bond[0]] = "\\"
            bond_order[bond[1]] = "/"
            tmol.GetBondWithIdx(bond[0]).SetBondDir(BondDir.ENDDOWNRIGHT)
            tmol.GetBondWithIdx(bond[1]).SetBondDir(BondDir.ENDUPRIGHT)

        for bond in bonds_set_trans:
            bond_order[bond[0]] = "/"
            bond_order[bond[1]] = "/"
            tmol.GetBondWithIdx(bond[0]).SetBondDir(BondDir.ENDUPRIGHT)
            tmol.GetBondWithIdx(bond[1]).SetBondDir(BondDir.ENDUPRIGHT)

        rdkit.Chem.rdmolops.FastFindRings(tmol)
        rdkit.Chem.rdmolops.SetBondStereoFromDirections(tmol)

        for bond in tmol.GetBonds():
            true_stereo = self.bond_indices_to_stereo[
                (
                    tmol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum(),
                    tmol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum(),
                )
            ]
            assigned_stereo = bond.GetStereo()

            if true_stereo != assigned_stereo:
                start_atom = tmol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                for nene in start_atom.GetNeighbors():
                    target_bond = tmol.GetBondBetweenAtoms(
                        start_atom.GetIdx(), nene.GetIdx()
                    )
                    if target_bond.GetBondDir() == BondDir.ENDUPRIGHT:
                        target_bond.SetBondDir(BondDir.ENDDOWNRIGHT)
                        bond_order[target_bond.GetIdx()] = "\\"
                        break
                    elif target_bond.GetBondDir() == BondDir.ENDDOWNRIGHT:
                        target_bond.SetBondDir(BondDir.ENDUPRIGHT)
                        bond_order[target_bond.GetIdx()] = "/"
                        break

        for bond in tmol.GetBonds():
            bond_dir = bond.GetBondDir()
            if bond_dir == BondDir.ENDDOWNRIGHT:
                sm_loc = bond_map[
                    (
                        tmol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum(),
                        tmol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum(),
                    )
                ]
                smarts_in = self.replace_at_index(smarts_in, "\\", sm_loc - 1, 1)

                # for b in bond_map:
                #     if bond_map[b] > sm_loc:
                #         bond_map[b] = bond_map[b] + 1
                # for a in node_map:
                #     if node_map[a] > sm_loc:
                #         node_map[a] = node_map[a] + 1

            elif bond_dir == BondDir.ENDUPRIGHT:
                sm_loc = bond_map[
                    (
                        tmol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum(),
                        tmol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum(),
                    )
                ]
                smarts_in = self.replace_at_index(smarts_in, "/", sm_loc - 1, 1)

        cw_ord = [0, 1, 2]
        ccw_ord = [0, 2, 1]

        flipped_chiral_tag = []
        for i, atm in enumerate(tmol.GetAtoms()):
            flipped_chiral_tag.append(atm.GetChiralTag())
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
                        # print(self.nodes[nn.GetAtomMapNum()])
                    else:
                        new_bond_indices.append(0)

                if len(old_bonds_indices) == 3:
                    old_bonds_indices.insert(1, -1)
                    new_bond_indices.insert(1, -1)
                # print(atm.GetAtomMapNum(), self.nodes[atm.GetAtomMapNum()])
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

        # CCW = @
        # CW = @@
        for i, r in enumerate(tmol.GetAtoms()):
            if str(r.GetChiralTag()) == "CHI_TETRAHEDRAL_CCW":
                sm = self.nodes[r.GetAtomMapNum()].data["smarts"]
                csm = re.sub(r.GetSymbol(), r.GetSymbol() + "@", sm)
                smarts_in = self.replace_at_index(
                    smarts_in, csm, node_map[r.GetAtomMapNum()] - len(sm), len(sm)
                )

                for nn in node_map:
                    if node_map[nn] > node_map[r.GetAtomMapNum()]:
                        node_map[nn] = node_map[nn] + 1
                for bb in bond_map:
                    if bond_map[bb] > node_map[r.GetAtomMapNum()]:
                        bond_map[bb] = bond_map[bb] + 1

            elif str(r.GetChiralTag()) == "CHI_TETRAHEDRAL_CW":
                sm = self.nodes[r.GetAtomMapNum()].data["smarts"]
                csm = re.sub(r.GetSymbol(), r.GetSymbol() + "@@", sm)

                smarts_in = self.replace_at_index(
                    smarts_in, csm, node_map[r.GetAtomMapNum()] - len(sm), len(sm)
                )

                for nn in node_map:
                    if node_map[nn] > node_map[r.GetAtomMapNum()]:
                        node_map[nn] = node_map[nn] + 2
                for bb in bond_map:
                    if bond_map[bb] > node_map[r.GetAtomMapNum()]:
                        bond_map[bb] = bond_map[bb] + 2

        smarts_in_no_map = re.sub(r":\d+]", "]", smarts_in)
        smarts_in_mapped = smarts_in
        self.unmapped_canon = smarts_in_no_map

        return smarts_in_no_map, smarts_in_mapped
