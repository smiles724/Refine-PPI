import torch

from ._base import register_transform


@register_transform('select_atom')
class SelectAtom(object):

    def __init__(self, resolution):
        super().__init__()
        assert resolution in ('full', 'backbone', 'backbone+CB')
        self.resolution = resolution

    def __call__(self, data):
        if self.resolution == 'full':
            data['pos_atoms'] = data['pos_heavyatom']
            data['type_atoms'] = data['type_heavyatom']
            data['mask_atoms'] = data['mask_heavyatom']
            data['bfactor_atoms'] = data['bfactor_heavyatom']
            data['pos_gt'] = data['pos_gt']
        elif self.resolution == 'backbone':
            data['pos_atoms'] = data['pos_heavyatom'][:, :4]
            data['type_atoms'] = data['type_heavyatom'][:, :4]
            data['mask_atoms'] = data['mask_heavyatom'][:, :4]
            data['bfactor_atoms'] = data['bfactor_heavyatom'][:, :4]
            data['pos_gt'] = data['pos_gt'][:, :4]
        elif self.resolution == 'backbone+CB':
            data['pos_atoms'] = data['pos_heavyatom'][:, :5]
            if 'type_heavyatom' in data.keys():
                data['type_atoms'] = data['type_heavyatom'][:, :5]
            else:
                data['type_atoms'] = torch.tensor([0, 1, 2, 3, 4]).int().repeat(len(data['pos_atoms']), 1)  # 'N', 'CA', 'C', 'O', 'CB'

            data['mask_atoms'] = data['mask_heavyatom'][:, :5]
            data['bfactor_atoms'] = data['bfactor_heavyatom'][:, :5]
            if 'pos_gt' in data.keys():
                data['pos_gt'] = data['pos_gt'][:, :5]
        return data
