import numpy as np
from dataset import ScoliosisDataset

IMG_DIR   = 'data/train/images/'
LABEL_DIR = 'data/train/labels/'

ds = ScoliosisDataset(IMG_DIR, LABEL_DIR)

all_c = []
for i in range(len(ds)):
    try:
        all_c.append(ds.get_centers(i))
    except Exception as e:
        print(f'  skip {i}: {e}')

all_c  = np.array(all_c)
mean_c = all_c.mean(axis=0)
std_c  = all_c.std(axis=0)

np.save('priors/mean_centers.npy', mean_c)
np.save('priors/std_centers.npy',  std_c)

print('\n=== POSITIONS MOYENNES (px) ===')
print(f'{"Vertebre":<10} {"X moy":<8} {"Y moy":<8} {"Y std":<8} Zone')
print('-'*46)
for i in range(17):
    z = 'HAUT' if i<6 else ('MILIEU' if i<12 else 'BAS')
    print(f'V{i+1:<9} {mean_c[i,0]:.1f}    {mean_c[i,1]:.1f}    +/-{std_c[i,1]:.1f}   {z}')

print('\n=== POINTS DE DEPART DES 3 AGENTS ===')
print(f'  HAUT   start_y = {mean_c[0,1]:.0f} px')
print(f'  MILIEU start_y = {mean_c[6,1]:.0f} px')
print(f'  BAS    start_y = {mean_c[12,1]:.0f} px')
print('\nPriors sauvegardes dans priors/')
