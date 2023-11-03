import torch

from ..protein import constants
from ._base import register_transform


@register_transform('merge_chains')
class MergeChains(object):

    def __init__(self, max_len=None, use_sasa=False):
        super().__init__()
        self.max_len = max_len
        self.use_sasa = use_sasa

    def assign_chain_number_(self, data_list):
        chains = set()
        for data in data_list:
            chains.update(data['chain_id'])
        chains = {c: i for i, c in enumerate(chains)}

        for data in data_list:
            data['chain_nb'] = torch.LongTensor([chains[c] for c in data['chain_id']])

    def _data_attr(self, data, name):
        return data[name]

    def __call__(self, structure):
        structure['ligand']['fragment_type'] = torch.full_like(structure['heavy']['aa'], fill_value=constants.Fragmentv2.Ligand, )
        structure['receptor']['fragment_type'] = torch.full_like(structure['light']['aa'], fill_value=constants.Fragmentv2.Receptor, )
        data_list = [structure['ligand'], structure['receptor']]

        self.assign_chain_number_(data_list)

        list_props = {'chain_id': [], 'icode': [], }
        tensor_props = {'chain_nb': [], 'resseq': [], 'res_nb': [], 'aa': [], 'pos_heavyatom': [], 'mask_heavyatom': [], 'fragment_type': [], }
        if self.use_sasa:
            tensor_props['sasa'] = []
        if 'plm_feature' in structure['ligand'].keys():
            tensor_props['plm_feature'] = []

        for data in data_list:
            for k in list_props.keys():
                attr = self._data_attr(data, k)
                if self.max_len is not None:
                    attr = attr[:self.max_len]
                list_props[k].append(attr)

            for k in tensor_props.keys():
                attr = self._data_attr(data, k)
                if self.max_len is not None:
                    attr = attr[:self.max_len]
                tensor_props[k].append(attr)

        list_props = {k: sum(v, start=[]) for k, v in list_props.items()}
        tensor_props = {k: torch.cat(v, dim=0) for k, v in tensor_props.items()}
        data_out = {**list_props, **tensor_props, 'id': structure['id']}
        return data_out
