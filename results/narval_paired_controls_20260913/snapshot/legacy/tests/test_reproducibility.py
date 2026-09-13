"""Reproductibilite du RNN et de la sauvegarde, sur de petites donnees synthetiques."""
import importlib.util
import pickle
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from maitrise_curbd.curbd import trainMultiRegionRNN

spec=importlib.util.spec_from_file_location('curbd_loop',ROOT/'scripts/curbd/loop.py')
loop=importlib.util.module_from_spec(spec)
spec.loader.exec_module(loop)


class ReproducibilityTests(unittest.TestCase):
    def setUp(self):
        time=np.arange(36)/12
        self.activity=np.stack([.3+.1*np.sin(time*(k+1)) for k in range(5)]).astype(np.float32)
        self.kw=dict(dtData=1/12,dtFactor=2,tauRNN=.33,g=.8,ampInWN=.01,
                     nRunTrain=4,nRunFree=2,plotStatus=False)

    def train(self,seed):
        return trainMultiRegionRNN(self.activity,seed=seed,**self.kw)

    def test_same_seed_is_exact_and_does_not_depend_on_global_rng(self):
        np.random.seed(77)
        a=self.train(2026)
        np.random.seed(99)
        b=self.train(2026)
        for key in ('J0','inputWN','J','RNN','pVars','iTarget'):
            np.testing.assert_array_equal(a[key],b[key])

    def test_explicit_seed_leaves_global_rng_untouched(self):
        np.random.seed(77)
        expected=np.random.random(5)
        np.random.seed(77)
        self.train(2026)
        np.testing.assert_array_equal(np.random.random(5),expected)

    def test_different_seeds_change_initialization_but_not_target(self):
        a,b=self.train(2026),self.train(2027)
        for key in ('J0','inputWN','J'):
            self.assertFalse(np.array_equal(a[key],b[key]))
        np.testing.assert_array_equal(a['Adata'],b['Adata'])

    def test_legacy_global_seed_matches_local_randomstate(self):
        np.random.seed(2026)
        a,b=self.train(None),self.train(2026)
        for key in ('J0','inputWN','J','RNN','pVars'):
            np.testing.assert_array_equal(a[key],b[key])

    def test_invalid_seeds_rejected(self):
        for seed in (-1,2**32,True,1.5,'2026'):
            with self.subTest(seed=seed),self.assertRaises(ValueError):
                self.train(seed)

    def test_training_duration_does_not_change_initialization(self):
        a=self.train(2026)
        b=trainMultiRegionRNN(self.activity,seed=2026,**{**self.kw,'nRunTrain':6})
        for key in ('J0','inputWN','iTarget'):
            np.testing.assert_array_equal(a[key],b[key])

    def prepared(self):
        return {'ts':self.activity, 'regions':np.array([['A',np.array([0,1])],['B',np.array([2,3,4])]],dtype=object),
                'n_subregions':5,'n_parent_regions':2,'duration_sec':3.,
                'masque_sub':np.arange(5).reshape(1,5),
                'info_masque_sub':{i:{'parent_region':0 if i<2 else 1,'n_pixels':1} for i in range(5)}}

    def test_save_reload_and_replay(self):
        params={**loop.BASE_PARAMS,**self.kw,'n_pixels':15,'seed':2026}
        params.pop('plotStatus')
        with tempfile.TemporaryDirectory() as tmp, patch.object(loop,'prepare_timeseries',return_value=self.prepared()):
            path=Path(tmp)/'model.pkl'
            row=loop.run_one_config(0,(9,6,410),params,None,path)
            self.assertEqual(row['status'],'done',row['error'])
            self.assertFalse(path.with_suffix('.pkl.tmp').exists())
            with path.open('rb') as f:
                data=pickle.load(f)
        expected=self.train(2026)
        for key in ('J0','inputWN','initial_state'):
            np.testing.assert_array_equal(data[key],expected[key])
        self.assertEqual(data['parameters']['seed'],2026)
        self.assertEqual(data['row']['seed'],2026)
        self.assertEqual(data['model_parameters']['seed'],2026)
        self.assertEqual(data['activity_scale'],float(self.activity.max()))
        self.assertIn('scripts/curbd/loop.py',data['reproducibility']['source_sha256'])
        self.assertEqual(data['reproducibility']['timeseries_sha256'],loop.reproducibility_metadata(self.activity)['timeseries_sha256'])
        np.testing.assert_array_equal(data['masque_sub'],self.prepared()['masque_sub'])
        self.assertEqual(data['info_masque_sub'],self.prepared()['info_masque_sub'])
        # Replay de la dynamique a poids fixes, independant de trainMultiRegionRNN.
        j,noise=data['J_final_full_precision'],data['inputWN']
        h=data['initial_state'][:,None].copy()
        replay=np.zeros_like(expected['RNN'])
        replay[:,0]=np.tanh(h[:,0])
        dt=params['dtData']/params['dtFactor']
        for tt in range(1,replay.shape[1]):
            replay[:,tt]=np.tanh(h[:,0])
            h=h+dt*(-h+j.dot(replay[:,tt]).reshape(-1,1)+noise[:,tt,None])/params['tauRNN']
        np.testing.assert_allclose(replay,expected['RNN'],rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(replay.astype(np.float32),data['RNN_final'])

    def test_seed_sweep_filenames_are_distinct(self):
        base={**loop.BASE_PARAMS,'n_pixels':15}
        configs=loop.build_sweep_configs(base,{'seed':[2026,2027,2028]})
        names=[loop.build_run_name(0,(9,6,410),p,['seed']) for p in configs]
        self.assertEqual(len(set(names)),3)
        self.assertTrue(all(len(n.encode())<=255 for n in names))
        self.assertTrue(all('segmentation_seed0' in n for n in names))

    def test_rnn_seed_does_not_change_segmentation_seed(self):
        prepared=self.prepared()
        tensor=self.activity.T[:,None,:]
        atlas=np.zeros((1,5))
        params={**loop.BASE_PARAMS,'n_pixels':15,'lissage_sigma':0}
        with patch.object(loop,'load_dataset',return_value=(tensor,atlas,atlas)), \
             patch.object(loop,'remove_thin_label_artifacts',return_value=atlas), \
             patch.object(loop,'reduce_atlas_to_6_regions',return_value=atlas), \
             patch.object(loop,'subdivide_mask_by_spatial_clustering',return_value=(prepared['masque_sub'],prepared['info_masque_sub'])) as subdivide:
            a=loop.prepare_timeseries((9,6,410),None,{**params,'seed':2026})
            b=loop.prepare_timeseries((9,6,410),None,{**params,'seed':2027})
        self.assertEqual([call.kwargs['random_state'] for call in subdivide.call_args_list],[0,0])
        np.testing.assert_array_equal(a['ts'],b['ts'])
        np.testing.assert_array_equal(a['ts'],self.activity)

    def test_failed_pickle_write_preserves_previous_file(self):
        params={**loop.BASE_PARAMS,**self.kw,'n_pixels':15,'seed':2026}
        params.pop('plotStatus')
        with tempfile.TemporaryDirectory() as tmp, patch.object(loop,'prepare_timeseries',return_value=self.prepared()):
            path=Path(tmp)/'model.pkl'
            path.write_bytes(b'previous result')
            with patch.object(loop.pickle,'dump',side_effect=OSError('simulated disk error')):
                row=loop.run_one_config(0,(9,6,410),params,None,path)
            self.assertEqual(row['status'],'failed')
            self.assertEqual(path.read_bytes(),b'previous result')
            self.assertFalse(path.with_suffix('.pkl.tmp').exists())


if __name__=='__main__':
    unittest.main()
