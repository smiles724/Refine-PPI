import random
import torch

from ._base import _index_select_data, register_transform, _get_CB_positions


@register_transform('focused_random_patch')
class FocusedRandomPatch(object):

    def __init__(self, focus_attr, seed_nbh_size=32, patch_size=128):
        super().__init__()
        self.focus_attr = focus_attr
        self.seed_nbh_size = seed_nbh_size
        self.patch_size = patch_size

    def __call__(self, data):
        focus_flag = (data[self.focus_attr] > 0)  # (L, )
        if focus_flag.sum() == 0:
            # If there is no active residues, randomly pick one.
            focus_flag[random.randint(0, focus_flag.size(0) - 1)] = True
        seed_idx = torch.multinomial(focus_flag.float(), num_samples=1).item()   # select a random residue

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, )
        pos_seed = pos_CB[seed_idx:seed_idx + 1]  # (1, )
        dist_from_seed = torch.cdist(pos_CB, pos_seed)[:, 0]  # (L, 1) -> (L, )
        nbh_seed_idx = dist_from_seed.argsort()[:self.seed_nbh_size]  # (Nb, )

        core_idx = nbh_seed_idx[focus_flag[nbh_seed_idx]]  # (Ac, ), the core-set must be a subset of the focus-set
        dist_from_core = torch.cdist(pos_CB, pos_CB[core_idx]).min(dim=1)[0]  # (L, )
        patch_idx = dist_from_core.argsort()[:self.patch_size]  # (P, )
        patch_idx = patch_idx.sort()[0]

        core_flag = torch.zeros([data['aa'].size(0), ], dtype=torch.bool)
        core_flag[core_idx] = True
        data['core_flag'] = core_flag

        data_patch = _index_select_data(data, patch_idx)
        return data_patch


@register_transform('random_patch')
class RandomPatch(object):

    def __init__(self, seed_nbh_size=32, patch_size=128):
        super().__init__()
        self.seed_nbh_size = seed_nbh_size
        self.patch_size = patch_size

    def __call__(self, data):
        seed_idx = random.randint(0, data['aa'].size(0) - 1)

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, )
        pos_seed = pos_CB[seed_idx:seed_idx + 1]  # (1, )
        dist_from_seed = torch.cdist(pos_CB, pos_seed)[:, 0]  # (L, 1) -> (L, )
        core_idx = dist_from_seed.argsort()[:self.seed_nbh_size]  # (Nb, )

        dist_from_core = torch.cdist(pos_CB, pos_CB[core_idx]).min(dim=1)[0]  # (L, )
        patch_idx = dist_from_core.argsort()[:self.patch_size]  # (P, )
        patch_idx = patch_idx.sort()[0]

        core_flag = torch.zeros([data['aa'].size(0), ], dtype=torch.bool)
        core_flag[core_idx] = True
        data['core_flag'] = core_flag

        data_patch = _index_select_data(data, patch_idx)
        return data_patch


@register_transform('selected_region_with_padding_patch')
class SelectedRegionWithPaddingPatch(object):

    def __init__(self, select_attr, each_residue_nbh_size, patch_size_limit):
        super().__init__()
        self.select_attr = select_attr
        self.each_residue_nbh_size = each_residue_nbh_size
        self.patch_size_limit = patch_size_limit

    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, 3)
        pos_sel = pos_CB[select_flag]  # (S, 3)
        dist_from_sel = torch.cdist(pos_CB, pos_sel)  # (L, S)
        nbh_sel_idx = torch.argsort(dist_from_sel, dim=0)[:self.each_residue_nbh_size, :]  # (nbh, S)
        patch_idx = nbh_sel_idx.view(-1).unique()  # (patchsize,)

        data_patch = _index_select_data(data, patch_idx)
        return data_patch


@register_transform('selected_region_fixed_size_patch')
class SelectedRegionFixedSizePatch(object):

    def __init__(self, select_attr, patch_size):
        super().__init__()
        self.select_attr = select_attr
        self.patch_size = patch_size

    def __call__(self, data):
        select_flag = (data[self.select_attr] > 0)

        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, 3)
        pos_sel = pos_CB[select_flag]  # (S, 3)
        try:
            dist_from_sel = torch.cdist(pos_CB, pos_sel).min(dim=1)[0]  # (L, )
        except:
            print(pos_CB.shape, pos_sel.shape)
        patch_idx = torch.argsort(dist_from_sel)[:self.patch_size]  # select at most patch_size residues that are closet to the+

        data_patch = _index_select_data(data, patch_idx)
        return data_patch


@register_transform('selected_interface_region_padding_patch')
class SelectedInterfaceRegionPaddingPatch(object):

    def __init__(self, cutoff, fix_size, fix_number):
        super().__init__()
        self.cutoff = cutoff
        self.fix_size = fix_size
        self.fix_number = fix_number

    def __call__(self, data):
        pos_CB = _get_CB_positions(data['pos_atoms'], data['mask_atoms'])  # (L, 3)
        ag_idx = [i for i in range(len(data['chain_id'])) if data['chain_id'][i] in data['ag_chain']]
        ab_idx = [i for i in range(len(data['chain_id'])) if data['chain_id'][i] not in data['ag_chain']]
        pos_ag = pos_CB[ag_idx]   # (AG, 3)
        pos_ab = pos_CB[ab_idx]   # (AB, 3)
        data.pop('ag_chain')

        dist = torch.cdist(pos_ag, pos_ab)  # (AG, AB)
        if self.fix_size:
            dist_from_ab = dist.min(dim=1)[0]   # (AG, )
            dist_from_ag = dist.min(dim=0)[0]   # (AB, )
            if self.fix_number < len(ag_idx):
                ag_id_selected = [ag_idx[i] for i in torch.topk(-dist_from_ab, self.fix_number)[1]]
            else:
                ag_id_selected = ag_idx
            if self.fix_number < len(ab_idx):
                ab_id_selected = [ab_idx[i] for i in torch.topk(-dist_from_ag, self.fix_number)[1]]
            else:
                ab_id_selected = ab_idx
        else:
            dist_cutoff = dist < self.cutoff
            ag_id_selected = [ag_idx[i] for i in range(len(ag_idx)) if torch.sum(dist_cutoff[i, :]) > 0]
            ab_id_selected = [ab_idx[i] for i in range(len(ab_idx)) if torch.sum(dist_cutoff[:, i]) > 0]
        assert len(ag_id_selected) > 0 and len(ab_id_selected) > 0, f'no residue has been selected from the complex structure. Min dist: {torch.min(dist)}. Enlarge the threshold!'
        data_patch = _index_select_data(data, ag_id_selected + ab_id_selected)
        return data_patch
