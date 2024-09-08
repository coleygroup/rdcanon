from rdkit import Chem
from rdkit.Chem import AllChem
import re
from rdcanon.token_parser import (
    order_token_canon,
    recursive_compare,
    parse_smarts_total,
)
import rdkit
from collections import deque
from rdcanon.askcos_prims import prims as prims1
import random
from functools import cmp_to_key
from rdkit.Chem.rdchem import BondType, BondDir, BondStereo
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

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

    if "unmapped_canon" in item1t:
        if item1t["unmapped_canon"] == item2t["unmapped_canon"]:
            sm1 = item1t["san_smarts"]
            sm2 = item2t["san_smarts"]
            return (sm1 > sm2) - (sm1 < sm2)

    return recursive_compare(item1, item2)


class Node:
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


class Graph:
    def __init__(self, v=False):
        self.nodes = []
        self.top_score = 0
        self.v = v
        self.bond_indices_to_smarts = {}
        self.bond_indices_to_stereo = {}
        self.bond_indices_to_relative_stereo = {}
        # self.atom_to_original_chiral_tag = {}

    def graph_from_smarts(self, smarts, embedding):
        proton_mol = Chem.MolFromSmiles("[#1]")
        
        mol = Chem.MolFromSmarts(smarts)
        
        if Chem.HasQueryHs(mol)[0]:
            mol = Chem.AdjustQueryProperties(Chem.MergeQueryHs(mol))
            # print(Chem.MolToSmarts(mol))
        if not mol:
            raise ValueError("Invalid SMARTS provided")

        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_NONE)

        # Create nodes for each atom
        if self.v:
            print("input SMARTS:", smarts)
            print("rd-sanitized SMARTS:", Chem.MolToSmarts(mol))
            print()
            print("token embeddings")

        num_atoms = len(mol.GetAtoms())
        atoms_seq, bonds_seq = parse_smarts_total(smarts, num_atoms)

        old_idx_to_new_idx = {}
        nnn = 0
        for atom in mol.GetAtoms():
            # is it supposed to make sense?
            if "@@" in atoms_seq[atom.GetIdx()]:
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW)
            elif "@" in atoms_seq[atom.GetIdx()]:
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW)

            if atom.GetSmarts() == "[H]" or atom.GetSmarts() == "[#1]":
                continue

            min_num_explicit_hs = 0
            opt_num_explicit_hs = 0
            # for neighbors in atom.GetNeighbors():
            #     neigh_sm = Chem.MolFromSmarts(neighbors.GetSmarts())
            #     if neighbors.GetSmarts() == "[H]" or neighbors.GetSmarts() == "[#1]":
            #         min_num_explicit_hs += 1
            #     elif proton_mol.HasSubstructMatch(neigh_sm):
            #         opt_num_explicit_hs += 1

            if min_num_explicit_hs == 0:
                min_num_explicit_hs = None
            if opt_num_explicit_hs == 0:
                opt_num_explicit_hs = None

            node_data = {
                "smarts": atom.GetSmarts(),
                "stereo": atom.GetChiralTag(),
            }
            n = Node(nnn, node_data)
            atom_map = re.findall(r":\d+]", n.data["smarts"])

            if len(atom_map) > 0:
                sm, sc, _ = order_token_canon(
                    re.sub(r":\d+]", "]", n.data["smarts"]),
                    atom_map[0][0:-1],
                    embedding,
                    min_num_explicit_hs,
                    opt_num_explicit_hs
                )
            else:
                sm, sc, _ = order_token_canon(
                    re.sub(r":\d+]", "]", n.data["smarts"]),
                    None,
                    embedding,
                    min_num_explicit_hs,
                    opt_num_explicit_hs
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
            old_idx_to_new_idx[atom.GetIdx()] = nnn
            nnn = nnn + 1
            self.nodes.append(n)
        if self.v:
            print()

        rdkit.Chem.rdmolops.FastFindRings(mol)
        rdkit.Chem.rdmolops.FindPotentialStereoBonds(mol)
        for bond in mol.GetBonds():
            if (
                bond.GetBeginAtomIdx() not in old_idx_to_new_idx
                or bond.GetEndAtomIdx() not in old_idx_to_new_idx
            ):
                continue
            start_idx_o = bond.GetBeginAtomIdx()
            end_idx_o = bond.GetEndAtomIdx()

            start_idx = old_idx_to_new_idx[bond.GetBeginAtomIdx()]
            end_idx = old_idx_to_new_idx[bond.GetEndAtomIdx()]

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

                atom = mol.GetAtomWithIdx(start_idx_o)
                neighbors = atom.GetNeighbors()

                for neighbor in neighbors:
                    if (
                        mol.GetBondBetweenAtoms(
                            start_idx_o, neighbor.GetIdx()
                        ).GetBondDir()
                        == Chem.rdchem.BondDir.ENDUPRIGHT
                        or mol.GetBondBetweenAtoms(
                            start_idx_o, neighbor.GetIdx()
                        ).GetBondDir()
                        == Chem.rdchem.BondDir.ENDDOWNRIGHT
                    ):
                        bond_a = neighbor.GetIdx()
                        break

                atom = mol.GetAtomWithIdx(end_idx_o)
                neighbors = atom.GetNeighbors()

                for neighbor in neighbors:
                    if (
                        mol.GetBondBetweenAtoms(
                            end_idx_o, neighbor.GetIdx()
                        ).GetBondDir()
                        == Chem.rdchem.BondDir.ENDUPRIGHT
                        or mol.GetBondBetweenAtoms(
                            end_idx_o, neighbor.GetIdx()
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
        nn = 0
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
            if nn > 10000:
                raise ValueError(
                    "Too many iterations, please share SMARTS with us as an issue on github"
                )
            nn = nn + 1
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
                                min([np, best_seen], key=cmp_to_key(recursive_compare))
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

                    if min([np, best_seen], key=cmp_to_key(recursive_compare)) == np:
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
        best_seen = []
        for idx, h in enumerate(self.nodes):
            if idx not in top_nodes:
                continue

            all_paths, new_best_seen, sm_o, node_map, bond_map = (
                self.find_hamiltonian_paths_iterative_sm(h.index, best_seen)
            )
            best_seen = new_best_seen

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
        self.unmapped_canon = sms[0][0]

        if mapping:
            return sms[0][1]
        else:
            return sms[0][0]

    def delete_at_index(self, original, start, length):
        end = start + length
        return original[:start] + original[end:]

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


        ###           ###
        ### Fix Bonds ###
        ###           ###

        bonds_set_equal = []
        bonds_set_trans = []
        for bond in tmol.GetBonds():
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
                # print(start_rs, end_rs)
                for i in start_rs:
                    for j in end_rs:
                        all_r_combos.append((i, j))

                # print(self.bond_indices_to_relative_stereo)
                # print(bond_map)
                # print(smarts_in)

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
                            # TODO: force cis (WIP)
                            if r[0] in start_rs:
                                v = r[0]
                                v2 = r[1]
                                # if len(start_rs) > 1:
                                #     for ea in start_rs:
                                #         if ea != v:
                                #             v = ea
                                #             break
                                # else:
                                #     for ea in end_rs:
                                #         if ea != v2:
                                #             smarts_in = self.insert_at_index(smarts_in, "-", node_map[ea]-len(self.nodes[ea].data["smarts"]))
                                #             print(bond_map)
                                #             for a in node_map:
                                #                 if node_map[a] > (node_map[ea]-len(self.nodes[ea].data["smarts"]) - 1):
                                #                     node_map[a] = node_map[a] + 1
                                #             for b in bond_map:
                                #                 if bond_map[b] > (node_map[ea]-len(self.nodes[ea].data["smarts"]) - 1):
                                #                     bond_map[b] = bond_map[b] + 1
                                #             print(bond_map)
                                #             smarts_in = self.delete_at_index(smarts_in, bond_map[(end_atom.GetAtomMapNum(), v2)]-1, 1)
                                #             for a in node_map:
                                #                 if node_map[a] > bond_map[(end_atom.GetAtomMapNum(), v2)]-1:
                                #                     node_map[a] = node_map[a] - 1
                                #             for b in bond_map:
                                #                 if bond_map[b] > bond_map[(end_atom.GetAtomMapNum(), v2)]-1:
                                #                     bond_map[b] = bond_map[b] - 1
                                #             bond_map[(end_atom.GetAtomMapNum(), ea)] = node_map[ea]-len(self.nodes[ea].data["smarts"])
                                #             print(bond_map)

                                #             v2 = ea
                                #             break
                                bond_a = tmol.GetBondBetweenAtoms(
                                    start_atom.GetIdx(), old_map_to_new_map[v]
                                ).GetIdx()
                                bond_b = tmol.GetBondBetweenAtoms(
                                    end_atom.GetIdx(), old_map_to_new_map[v2]
                                ).GetIdx()
                            else:
                                v = r[0]
                                v2 = r[1]
                                # if len(start_rs) > 1:
                                #     for ea in start_rs:
                                #         if ea != v:
                                #             v = ea
                                #             break
                                # else:
                                #     for ea in end_rs:
                                #         if ea != v2:
                                #             v2 = ea
                                #             break
                                bond_a = tmol.GetBondBetweenAtoms(
                                    start_atom.GetIdx(), old_map_to_new_map[v]
                                ).GetIdx()
                                bond_b = tmol.GetBondBetweenAtoms(
                                    end_atom.GetIdx(), old_map_to_new_map[v2]
                                ).GetIdx()
                            bonds_set_equal.append((bond_a, bond_b))

        # print(bond_map)

        # print(bonds_set_equal)
        # print(bonds_set_trans)

        for bond in bonds_set_equal:
            bond = sorted(bond)
            tmol.GetBondWithIdx(bond[0]).SetBondDir(BondDir.ENDDOWNRIGHT)
            tmol.GetBondWithIdx(bond[1]).SetBondDir(BondDir.ENDUPRIGHT)

        for bond in bonds_set_trans:
            bond = sorted(bond)
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

            # if true_stereo != BondStereo.STEREONONE:
            # true_stereo = BondStereo.STEREOCIS
            assigned_stereo = bond.GetStereo()

            if true_stereo != assigned_stereo:
                # print(assigned_stereo)
                start_atom = tmol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                for nene in start_atom.GetNeighbors():
                    target_bond = tmol.GetBondBetweenAtoms(
                        start_atom.GetIdx(), nene.GetIdx()
                    )
                    if target_bond.GetBondDir() == BondDir.ENDUPRIGHT:
                        target_bond.SetBondDir(BondDir.ENDDOWNRIGHT)
                        break
                    elif target_bond.GetBondDir() == BondDir.ENDDOWNRIGHT:
                        target_bond.SetBondDir(BondDir.ENDUPRIGHT)
                        break

        for bond in tmol.GetBonds():
            bond_dir = bond.GetBondDir()
            # print(tmol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetAtomMapNum(),
            #         tmol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetAtomMapNum(),
            #         bond_dir)
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

        ###    END    ###
        ### Fix Bonds ###
        ###           ###



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
                    else:
                        new_bond_indices.append(0)

                if len(old_bonds_indices) == 3:
                    old_bonds_indices.insert(1, -1)
                    new_bond_indices.insert(1, -1)

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

        return smarts_in_no_map, smarts_in_mapped

    def __repr__(self):
        return f"Graph({self.nodes})"


class Reaction:
    def __init__(
        self,
        input_reaction_smarts,
        mapping,
        embedding,
        remapping=False,
        v=False,
        repl_dict={},
    ):
        self.reactants = []
        self.agents = []
        self.products = []
        self.input_reaction_smarts = input_reaction_smarts
        self.mapping = mapping
        self.embedding = embedding
        self.remapping = remapping
        self.v = v
        self.index = 1
        self.index_map = {}
        self.repl_dict = repl_dict

    def _load_reactants(self, reactants):
        for r_sm in reactants:
            is_grouped = False
            if r_sm[0] == "(":
                r_sm = r_sm[1:-1]
                is_grouped = True
            else:
                r_sm = r_sm

            if "." in r_sm:
                smss = r_sm.split(".")
            else:
                smss = [r_sm]
            grouped = []
            for sm in smss:
                san_sm, ts, unmapped_canon = canon_smarts(
                    sm,
                    self.mapping,
                    self.embedding,
                    return_score=True,
                    repl_dict=self.repl_dict,
                )
                grouped.append(
                    {
                        "path_scores": ts,
                        "san_smarts": san_sm,
                        "unmapped_canon": unmapped_canon,
                    }
                )

            grouped = sorted(grouped, key=cmp_to_key(custom_key2))

            san_sm_out = ".".join([x["san_smarts"] for x in grouped])
            unmapped_san_sm_out = ".".join([x["unmapped_canon"] for x in grouped])
            tss = [x["path_scores"] for x in grouped]
            if is_grouped:
                san_sm_out = "(" + san_sm_out + ")"
                unmapped_san_sm_out = "(" + unmapped_san_sm_out + ")"

            self.reactants.append(
                {
                    "path_scores": tss,
                    "original_smarts": r_sm,
                    "san_smarts": san_sm_out,
                    "unmapped_canon": unmapped_san_sm_out,
                }
            )

    def _load_agents(self, agents):
        for r_sm in agents:
            is_grouped = False
            if r_sm[0] == "(":
                r_sm = r_sm[1:-1]
                is_grouped = True
            else:
                r_sm = r_sm

            if "." in r_sm:
                smss = r_sm.split(".")
            else:
                smss = [r_sm]
            grouped = []
            for sm in smss:
                san_sm, ts, unmapped_canon = canon_smarts(
                    sm,
                    self.mapping,
                    self.embedding,
                    return_score=True,
                    repl_dict=self.repl_dict,
                )
                grouped.append(
                    {
                        "path_scores": ts,
                        "san_smarts": san_sm,
                        "unmapped_canon": unmapped_canon,
                    }
                )

            grouped = sorted(grouped, key=cmp_to_key(custom_key2))
            san_sm_out = ".".join([x["san_smarts"] for x in grouped])
            unmapped_san_sm_out = ".".join([x["unmapped_canon"] for x in grouped])
            tss = [x["path_scores"] for x in grouped]
            if is_grouped:
                san_sm_out = "(" + san_sm_out + ")"
                unmapped_san_sm_out = "(" + unmapped_san_sm_out + ")"

            self.agents.append(
                {
                    "path_scores": tss,
                    "original_smarts": r_sm,
                    "san_smarts": san_sm_out,
                    "unmapped_canon": unmapped_san_sm_out,
                }
            )

    def _load_products(self, products):
        for r_sm in products:
            is_grouped = False
            if r_sm[0] == "(":
                r_sm = r_sm[1:-1]
                is_grouped = True
            else:
                r_sm = r_sm

            if "." in r_sm:
                smss = r_sm.split(".")
            else:
                smss = [r_sm]
            grouped = []
            for sm in smss:
                san_sm, ts, unmapped_canon = canon_smarts(
                    sm,
                    self.mapping,
                    self.embedding,
                    return_score=True,
                    repl_dict=self.repl_dict,
                )
                grouped.append(
                    {
                        "path_scores": ts,
                        "san_smarts": san_sm,
                        "unmapped_canon": unmapped_canon,
                    }
                )

            grouped = sorted(grouped, key=cmp_to_key(custom_key2))
            san_sm_out = ".".join([x["san_smarts"] for x in grouped])
            unmapped_san_sm_out = ".".join([x["unmapped_canon"] for x in grouped])
            tss = [x["path_scores"] for x in grouped]
            if is_grouped:
                san_sm_out = "(" + san_sm_out + ")"
                unmapped_san_sm_out = "(" + unmapped_san_sm_out + ")"

            self.products.append(
                {
                    "path_scores": tss,
                    "original_smarts": r_sm,
                    "san_smarts": san_sm_out,
                    "unmapped_canon": unmapped_san_sm_out,
                }
            )

    def remap(self, in_smarts):
        for r in in_smarts:
            grouped = False
            if r["san_smarts"][0] == "(":
                t_sm = r["san_smarts"][1:-1]
                grouped = True
            else:
                t_sm = r["san_smarts"]
            sm_mol = Chem.MolFromSmarts(t_sm)
            replacements = {}
            for a in sm_mol.GetAtoms():
                if a.GetAtomMapNum() == 0:
                    continue
                if a.GetAtomMapNum() not in self.index_map:
                    self.index_map[a.GetAtomMapNum()] = self.index
                    self.index = self.index + 1
                replacements[":" + str(a.GetAtomMapNum()) + "]"] = (
                    ":" + str(self.index_map[a.GetAtomMapNum()]) + "]"
                )

            pattern = re.compile(
                "|".join(re.escape(key) for key in replacements.keys())
            )
            result = pattern.sub(lambda match: replacements[match.group(0)], t_sm)

            r["san_smarts"] = result
            if grouped:
                r["san_smarts"] = "(" + r["san_smarts"] + ")"

    def split_at_period_not_in_parentheses(self, s):
        parts = []
        current_part = []
        depth = 0  # Track the depth of parentheses nesting

        for char in s:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            elif char == "." and depth == 0:
                # If we're not inside parentheses, treat the period as a delimiter
                parts.append("".join(current_part))
                current_part = []
                continue

            # Add the current character to the part we're building
            current_part.append(char)

        # Add the last part, if there is one
        if current_part:
            parts.append("".join(current_part))

        return parts

    def canonicalize_template(self):
        k = AllChem.ReactionFromSmarts(self.input_reaction_smarts)
        if len(k.GetAgents()) > 0:
            comps = self.input_reaction_smarts.split(">")
            if len(comps) == 3:
                reactants = self.split_at_period_not_in_parentheses(comps[0])
                agents = self.split_at_period_not_in_parentheses(comps[1])
                products = self.split_at_period_not_in_parentheses(comps[2])
            else:
                reactants = self.split_at_period_not_in_parentheses(comps[0])
                agents = []
                products = self.split_at_period_not_in_parentheses(comps[1])
        else:
            comps = self.input_reaction_smarts.split(">>")
            reactants = self.split_at_period_not_in_parentheses(comps[0])
            agents = []
            products = self.split_at_period_not_in_parentheses(comps[1])

        self._load_reactants(reactants)
        self._load_agents(agents)
        self._load_products(products)

        reactants_sort = sorted(self.reactants, key=cmp_to_key(custom_key2))

        if self.remapping:
            self.remap(reactants_sort)

        san_smarts_out = ".".join([r["san_smarts"] for r in reactants_sort])

        if len(self.agents) > 0:
            san_smarts_out = san_smarts_out + ">"
            agents_sort = sorted(self.agents, key=cmp_to_key(custom_key2))
            if self.remapping:
                self.remap(agents_sort)
            san_smarts_out = san_smarts_out + ".".join(
                [r["san_smarts"] for r in agents_sort]
            )

        san_smarts_out = san_smarts_out + ">>"
        products_sort = sorted(self.products, key=cmp_to_key(custom_key2))
        if self.remapping:
            self.remap(products_sort)
        san_smarts_out = san_smarts_out + ".".join(
            [r["san_smarts"] for r in products_sort]
        )

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
    smarts,
    mapping=False,
    embedding="drugbank",
    return_score=False,
    v=False,
    repl_dict={},
):
    """
    Canonicalizes a SMARTS pattern.

    Args:
        smarts (str): The input SMARTS pattern to be canonicalized.
        mapping (bool, optional): Whether to return the atom mapping. Defaults to False.
        embedding (str, optional): The query primitive frequency dictionary to use. Defaults to "drugbank".
        return_score (bool, optional): Whether to return the top score. Defaults to False.
        v (bool, optional): Whether to enable verbose mode. Defaults to False.
        repl_dict (dictionary, optional): A dictionary of SMARTS token replacements.

    Returns:
        str or tuple: The canonicalized SMARTS pattern. If `return_score` is True, a tuple containing the canonicalized SMARTS pattern,
        the top score, and the unmapped canonical SMARTS pattern is returned.
    """
    g = Graph(v)
    g.graph_from_smarts(smarts, embedding)
    out = g.recreate_molecule(mapping)

    for k in repl_dict:
        out = out.replace(k, repl_dict[k])

    if return_score:
        return out, g.top_score, g.unmapped_canon
    return out


def gen_canon_repl_dict(repl_dict, embedding="drugbank"):
    """
    Generate a canonical replacement dictionary based on a given replacement dictionary.

    Parameters:
        repl_dict (dict): A dictionary containing the replacement mappings.
        embedding (str, optional): The embedding to be used for generating canonical SMILES. Defaults to "drugbank".

    Returns:
        dict: A dictionary containing the canonical replacement mappings.
    """
    repl_dict_nodes = {}
    for k in repl_dict:
        repl_dict_nodes[canon_smarts(k, embedding)[1:-1]] = canon_smarts(
            repl_dict[k], embedding
        )[1:-1]
    return repl_dict_nodes


def debug(smarts, mapping=False, embedding="drugbank", return_score=False):
    canon_smarts(smarts, mapping, embedding, return_score, True)


def canon_reaction_smarts(
    smarts, mapping=False, embedding="drugbank", remapping=False, repl_dict={}
):
    """
    Canonicalizes a reaction SMARTS string.

    Args:
        smarts (str): The reaction SMARTS string to be canonicalized.
        mapping (bool, optional): Whether to include atom mapping in the canonicalization. Defaults to False.
        embedding (str, optional): The embedding to use for the canonicalization. Defaults to "drugbank".
        remapping (bool, optional): Whether to remap atom indices after canonicalization. Defaults to True.
        repl_dict (dictionary, optional): A dictionary of SMARTS token replacements.

    Returns:
        str: The canonicalized reaction SMARTS string.
    """

    if remapping == True:
        mapping = True

    reaction = Reaction(smarts, mapping, embedding, remapping, repl_dict=repl_dict)
    return reaction.canonicalize_template()
