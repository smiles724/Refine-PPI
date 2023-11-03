import os
from tqdm import tqdm

sasa_path = 'data/SKEMPI_v2_cache/SASA/'
os.makedirs(sasa_path, exist_ok=True)

chain_list = [i for i in os.listdir('./data/SKEMPI_v2/PDBs/') if i.endswith('.pdb') and not i.startswith('.')]
for pdb in tqdm(chain_list):
    pdb = pdb.split('.')[0]
    if not os.path.exists(os.path.join(sasa_path, pdb)):
        try:
            os.system('python sasa_mdtraj.py ./data/SKEMPI_v2/PDBs/' + pdb + '.pdb data/SKEMPI_v2_cache/SASA/' + pdb)
        except:
            print(f'{pdb} cannot be processed...')
            continue



