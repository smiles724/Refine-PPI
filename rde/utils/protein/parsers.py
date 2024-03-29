import torch
from Bio.PDB import Selection
from Bio.PDB.Residue import Residue
from easydict import EasyDict

from .constants import (AA, max_num_heavyatoms, restype_to_heavyatom_names, BBHeavyAtom, HeavyAtom2int)
from .icoord import get_chi_angles, get_backbone_torsions


def _get_residue_heavyatom_info(res: Residue):
    pos_heavyatom = torch.zeros([max_num_heavyatoms, 3], dtype=torch.float)
    type_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.int)
    mask_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.bool)
    bfactor_heavyatom = torch.zeros([max_num_heavyatoms, ], dtype=torch.float)
    restype = AA(res.get_resname())
    for idx, atom_name in enumerate(restype_to_heavyatom_names[restype]):
        if atom_name == '': continue
        if atom_name in res:
            pos_heavyatom[idx] = torch.tensor(res[atom_name].get_coord().tolist(), dtype=pos_heavyatom.dtype)
            type_heavyatom[idx] = torch.tensor(HeavyAtom2int.get(res[atom_name].get_name(), 6), dtype=type_heavyatom.dtype)
            mask_heavyatom[idx] = True
            bfactor_heavyatom[idx] = res[atom_name].get_bfactor()
    return pos_heavyatom, type_heavyatom, mask_heavyatom, bfactor_heavyatom


def parse_biopython_structure(entity, interface_idx=None, sasa=None, unknown_threshold=1.0, name=None):
    chains = Selection.unfold_entities(entity, 'C')
    chains.sort(key=lambda c: c.get_id())
    data = EasyDict(
        {'chain_id': [], 'chain_nb': [], 'resseq': [],  'icode': [], 'res_nb': [], 'aa': [], 'pos_heavyatom': [], 'type_heavyatom': [],
         'mask_heavyatom': [], 'bfactor_heavyatom': [], 'phi': [], 'phi_mask': [], 'psi': [], 'psi_mask': [], 'chi': [], 'chi_alt': [], 'chi_mask': [], 'chi_complete': [], })
    tensor_types = {'chain_nb': torch.LongTensor, 'resseq': torch.LongTensor, 'res_nb': torch.LongTensor, 'aa': torch.LongTensor,
                    'pos_heavyatom': torch.stack, 'mask_heavyatom': torch.stack, 'bfactor_heavyatom': torch.stack, 'type_heavyatom': torch.stack,
                    'phi': torch.FloatTensor, 'phi_mask': torch.BoolTensor, 'psi': torch.FloatTensor, 'psi_mask': torch.BoolTensor,
                    'chi': torch.stack, 'chi_alt': torch.stack, 'chi_mask': torch.stack, 'chi_complete': torch.BoolTensor, }
    if sasa is not None and len(sasa) > 0:
        data['sasa'] = []
        tensor_types['sasa'] = torch.LongTensor

    count_aa, count_unk = 0, 0
    for i, chain in enumerate(chains):
        chain.atom_to_internal_coordinates()  # biopython >= 1.81, compute bond lengths, angles, dihedral angles

        seq_this = 0  # Renumbering residues
        residues = Selection.unfold_entities(chain, 'R')
        residues.sort(key=lambda res: (res.get_id()[1], res.get_id()[2]))  # Sort residues by resseq-icode

        for _, res in enumerate(residues):
            if interface_idx is not None:
                if int(res.get_id()[1]) not in interface_idx:
                    continue

            resname = res.get_resname()
            if not AA.is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue
            restype = AA(resname)
            count_aa += 1
            if restype == AA.UNK or resname == 'UNK':
                count_unk += 1
                continue
            data.aa.append(restype)  # Will be automatically cast to torch.long

            # Chain info
            data.chain_id.append(chain.get_id())
            data.chain_nb.append(i)

            # Heavy atoms
            pos_heavyatom, type_heavyatom, mask_heavyatom, bfactor_heavyatom = _get_residue_heavyatom_info(res)
            data.pos_heavyatom.append(pos_heavyatom)
            data.type_heavyatom.append(type_heavyatom)
            data.mask_heavyatom.append(mask_heavyatom)
            data.bfactor_heavyatom.append(bfactor_heavyatom)

            # Backbone torsions
            phi, psi, _ = get_backbone_torsions(res)
            if phi is None:
                data.phi.append(0.0)
                data.phi_mask.append(False)
            else:
                data.phi.append(phi)
                data.phi_mask.append(True)
            if psi is None:
                data.psi.append(0.0)
                data.psi_mask.append(False)
            else:
                data.psi.append(psi)
                data.psi_mask.append(True)

            # Chi
            chi, chi_alt, chi_mask, chi_complete = get_chi_angles(restype, res)
            data.chi.append(chi)
            data.chi_alt.append(chi_alt)
            data.chi_mask.append(chi_mask)
            data.chi_complete.append(chi_complete)

            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]    # different state in the same residue number
            if seq_this == 0:
                seq_this = 1
            else:
                d_CA_CA = torch.linalg.norm(data.pos_heavyatom[-2][BBHeavyAtom.CA] - data.pos_heavyatom[-1][BBHeavyAtom.CA], ord=2).item()
                if d_CA_CA <= 4.0:
                    seq_this += 1
                else:
                    d_resseq = resseq_this - data.resseq[-1]
                    seq_this += max(2, d_resseq)

            data.resseq.append(resseq_this)
            data.icode.append(icode_this)
            data.res_nb.append(seq_this)    # renumbering index

            if 'sasa' in data:
                try:
                    data.sasa.append(sasa[chain.get_id() + resname + str(res.get_id()[1])])
                except:
                    data.sasa.append(0.0)
                    print(f'Warning: no sasa feature for {chain.get_id() + resname + str(res.get_id()[1])}')

    if len(data.aa) == 0 or (count_unk / count_aa) >= unknown_threshold:
        print(f'Warning: {name} has {len(data.aa)} residues with {(count_unk / count_aa) if count_aa > 0 else "high" } unknown rate. ')
        return None, None

    seq_map = {}       # (chain_id, resseq, icode) can be the same for different residues (2FTL), so len(seq_map) != len(data.pos_heavyatom)
    for i, (chain_id, resseq, icode) in enumerate(zip(data.chain_id, data.resseq, data.icode)):
        if (chain_id, resseq, icode) in seq_map.keys():
            print(f'Warning: {name} -- { (chain_id, resseq, icode)}')
            continue
        seq_map[(chain_id, resseq, icode)] = i

    for key, convert_fn in tensor_types.items():
        data[key] = convert_fn(data[key])
    return data, seq_map
