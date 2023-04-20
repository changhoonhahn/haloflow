import os, sys
import h5py, glob
import numpy as np
from tqdm.notebook import tqdm, trange
from astropy.table import Table, vstack

##################################################################################
snapshot = int(sys.argv[1]) 
##################################################################################

dat_dir = '/scratch/gpfs/chhahn/haloflow/'
grp_dir = '/scratch/gpfs/chhahn/haloflow/groupcat/idark.ipmu.jp/hsc405/GroupCats/groups_0%i/' % snapshot

h = 0.6773358287273804

##################################################################################
# read subhalos 
subhalo = Table.read(os.path.join(dat_dir, 'subhalos_morph.csv'))
is_snap  = (subhalo['snapshot'] == snapshot)
subhalo = subhalo[is_snap]
print('%i subhalos' % len(subhalo)) 


# compile subhalos and groups
tab_sub, tab_grp = [], []
for i in range(np.max([int(fsub.split('.')[-2]) for fsub in glob.glob(os.path.join(grp_dir, '*.hdf5'))])+1):
    with h5py.File(os.path.join(grp_dir, 'fof_subhalo_tab_0%i.%i.hdf5' % (snapshot, i)), 'r') as fsub:
        _tab = Table()
        for k in fsub['Subhalo'].keys():
            _tab[k] = fsub['Subhalo'][k][...]
        tab_sub.append(_tab)

        _tab = Table()
        for k in fsub['Group'].keys():
            _tab[k] = fsub['Group'][k][...]
        tab_grp.append(_tab)

tab_sub = vstack(tab_sub)
tab_grp = vstack(tab_grp)
##################################################################################
# get centrals
central_subid = tab_grp['GroupFirstSub'][tab_grp['GroupFirstSub'] != -1]
is_central = np.array([_id in central_subid for _id in subhalo['subhalo_id']])
print('%i centrals out of %i subhalos' % (np.sum(is_central), len(is_central)))

subhalo = subhalo[is_central]

# compile satellite luminosities 
lum_has_stars = np.zeros((len(subhalo), 4))
lum_above_mlim = np.zeros((len(subhalo), 4))
richness_all = np.zeros(len(subhalo))
richness_mlim = np.zeros(len(subhalo))

has_stars = tab_sub['SubhaloMassType'][:,4] > 0
above_mlim = np.log10(tab_sub['SubhaloMassType'][:,4] * 10**10 / h) > 9.

for i_sub in tqdm(np.unique(subhalo['subhalo_id'])):
    i_grp = tab_sub['SubhaloGrNr'][i_sub]
    in_group = (tab_sub['SubhaloGrNr'] == i_grp) & (np.arange(len(tab_sub)) != i_sub)

    # g, r, i, z
    is_sub = (subhalo['subhalo_id'] == i_sub)
    lum_has_stars[is_sub,:] = np.tile(
        np.sum(10**(-0.4 * tab_sub[in_group & has_stars]['SubhaloStellarPhotometrics'][:,4:]), 
            axis=0),
        (np.sum(is_sub),1))
    lum_above_mlim[is_sub,:] = np.tile(
        np.sum(10**(-0.4 * tab_sub[in_group & above_mlim]['SubhaloStellarPhotometrics'][:,4:]), 
            axis=0),
        (np.sum(is_sub),1))

    richness_all[is_sub]    = np.sum(in_group & has_stars)
    richness_mlim[is_sub]   = np.sum(in_group & above_mlim)

for i, b in enumerate(['g', 'r', 'i', 'z']): 
    subhalo['%s_lum_has_stars' % b] = lum_has_stars[:,i]
    subhalo['%s_lum_above_mlim' % b] = lum_above_mlim[:,i]

subhalo['richness_all'] = richness_all
subhalo['richness_mlim'] = richness_mlim

subhalo.write(os.path.join(dat_dir, 'subhalos.central.snapshot%i.hdf5' % snapshot), overwrite=True)

# set up training and test data sets 
uid = np.random.choice(np.unique(subhalo['subhalo_id'][subhalo['SubhaloMassType_stars'] > 9.5]), replace=False, size=125)

i_test = np.zeros(len(subhalo)).astype(bool)
for _uid in uid:
    i_test[subhalo['subhalo_id'] == _uid] = True

print('%s test subhalos' % np.sum(i_test))
test_subhalos = subhalo[i_test]
train_subhalos = subhalo[~i_test]

test_subhalos.write(os.path.join(dat_dir, 'subhalos.central.snapshot%i.test.hdf5' % snapshot), overwrite=True)
train_subhalos.write(os.path.join(dat_dir, 'subhalos.central.snapshot%i.train.hdf5' % snapshot), overwrite=True)
