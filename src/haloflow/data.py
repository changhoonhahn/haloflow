'''

module to interface with simulation data


'''
import os 
import numpy as np 
from astropy.table import Table 


if os.environ['machine'] == 'della': 
    dat_dir = '/scratch/gpfs/chhahn/haloflow/'
else: 
    raise ValueError


def get_subhalos(dataset, obs, snapshot=91): 
    ''' see nb/compile_subhalos.ipynb and nb/datasets.ipynb
    '''
    if snapshot != 91: raise NotImpelmentedError  
    fdata  = os.path.join(dat_dir, 'subhalos.central.snapshot%i.%s.csv' % (snapshot, dataset))
    
    if os.path.isfile(fdata): 
        subhalo   = Table.read(fdata)
    else: 
        subhalo = Table.read(os.path.join(dat_dir, 'subhalos_morph.csv'))
        subhalo = subhalo[subhalo['snapshot'] == snapshot]
        print('%i subhalos' % len(subhalo))
    
        central_id = np.load(os.path.join(dat_dir, 'centrals.subfind_id.snapshot%i.npy' % snapshot))

        is_central = np.array([_id in central_id for _id in subhalo['subhalo_id']])
        subhalo = subhalo[is_central]
        print('%.2f of subhalos are centrals' % np.mean(is_central))
        print('%i subhalos' % len(subhalo))

        
        uid = np.random.choice(np.unique(subhalo['subhalo_id'][subhalo['SubhaloMassType_stars'] > 9.5]), replace=False, size=125)

        i_test = np.zeros(len(subhalo)).astype(bool)
        for _uid in uid:
            i_test[subhalo['subhalo_id'] == _uid] = True
        
        subhalo_test = subhalo[i_test]
        subhalo_train = subhalo[~i_test]

        ftrain  = os.path.join(dat_dir, 'subhalos.central.snapshot%i.train.csv' % snapshot)
        ftest  = os.path.join(dat_dir, 'subhalos.central.snapshot%i.test.csv' % snapshot)
        subhalo_test.write(ftest) 
        subhalo_train.write(ftrain)
        
        if dataset == 'train': 
            subhalo = subhalo_train 
        elif dataset == 'test': 
            subhalo = subhalo_test

    if obs == 'mags':
        props = ['Sersic_mag'] 
    elif obs == 'mags_morph': 
        props = ['Sersic_Reff', 'Sersic_mag', 'CAS_C', 'CAS_A']
    else: 
        raise NotImplementedError

    cols = []
    for b in ['g', 'r', 'i', 'y', 'z']: 
        for p in props: 
            cols.append('%s_%s' % (b, p))

    y_train = np.array([np.array(subhalo[col].data) for col in ['SubhaloMassType_stars', 'SubhaloMassType_dm']]).T # stellar and halo mass 
    x_train = np.array([np.array(subhalo[col].data) for col in cols]).T

    return y_train, x_train 
