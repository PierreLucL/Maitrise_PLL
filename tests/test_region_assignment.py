"""Garde-fous sur les erreurs qui pourraient changer l'interprétation régionale."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from maitrise_curbd.masks import build_parent_regions_dict,validate_region_assignment
from maitrise_curbd.timeseries import extract_timeseries_du_tenseur

class AssignmentTests(unittest.TestCase):
    def setUp(self):
        self.parent=np.array([[0,0,1],[0,np.nan,1.]])
        self.mask=np.array([[0,0,1],[2,np.nan,1.]])
        self.info={0:dict(parent_region=0,n_pixels=2),1:dict(parent_region=1,n_pixels=2),2:dict(parent_region=0,n_pixels=1)}
        self.regions=build_parent_regions_dict(self.info)
    def check(self):
        return validate_region_assignment(self.parent,self.mask,self.info,self.regions,3)
    def test_rows_match_known_pixel_means(self):
        movie=np.array([[[2,4,10],[8,999,14]],[[4,6,20],[12,999,24]]])
        traces=extract_timeseries_du_tenseur(movie,self.mask)
        np.testing.assert_array_equal(traces,[[3,5],[12,22],[8,12]])
        self.assertEqual(self.check()['singleton_ids'],[2])
        np.testing.assert_array_equal(traces[self.regions[0,1]],[[3,5],[8,12]])
    def test_wrong_parent_rejected(self):
        self.info[1]['parent_region']=0
        with self.assertRaises(ValueError):self.check()
    def test_duplicate_or_missing_id_rejected(self):
        self.regions[0,1]=np.array([0,0])
        with self.assertRaises(ValueError):self.check()
    def test_renamed_row_rejected(self):
        self.regions[0,0]='Rég. Vis.'
        with self.assertRaises(ValueError):self.check()
    def test_added_background_pixel_rejected(self):
        self.mask[1,1]=0
        with self.assertRaises(ValueError):self.check()
    def test_sparse_ids_rejected(self):
        self.mask[self.mask==2]=3
        with self.assertRaises(ValueError):self.check()
    def test_disconnected_is_reported(self):
        self.mask[1,0]=0;self.mask[0,0]=2
        self.info[0]['n_pixels']=2
        self.assertEqual(self.check()['disconnected_ids_4_neighbors'],[0])

if __name__=='__main__':unittest.main()

class RepairTests(unittest.TestCase):
    def test_remote_thin_label_corrected_without_filling_background(self):
        from maitrise_curbd.masks import repair_anatomical_outliers
        a=np.zeros((20,30));a[:,20:]=1;a[8,3:10]=1;a[0,0]=np.nan
        b,q=repair_anatomical_outliers(a,np.ones_like(a,bool))
        np.testing.assert_array_equal(b[8,3:10],0)
        np.testing.assert_array_equal(b[:,20:],1)
        self.assertTrue(np.isnan(b[0,0]));self.assertEqual(q['changed_pixels'],7)
    def test_no_core_does_not_invent_labels(self):
        from maitrise_curbd.masks import repair_anatomical_outliers
        a=np.full((9,9),np.nan);a[4,:]=2
        b,q=repair_anatomical_outliers(a,np.ones_like(a,bool))
        np.testing.assert_array_equal(a,b);self.assertEqual(q['parents_without_core'],[2])
    def test_empty_atlas_rejected(self):
        from maitrise_curbd.masks import repair_anatomical_outliers
        with self.assertRaises(ValueError):repair_anatomical_outliers(np.full((4,4),np.nan),np.ones((4,4)))
    def test_connected_parcels_preserve_parents_and_isolated_pixel(self):
        from maitrise_curbd.masks import make_connected_subgroups
        p=np.array([[0,0,1,1],[0,0,1,1],[np.nan,np.nan,np.nan,np.nan],[0,np.nan,np.nan,np.nan]])
        m=np.array([[0,1,2,2],[1,0,2,3],[np.nan,np.nan,np.nan,np.nan],[0,np.nan,np.nan,np.nan]])
        b,info=make_connected_subgroups(p,m);regions=build_parent_regions_dict(info)
        q=validate_region_assignment(p,b,info,regions,len(info))
        self.assertEqual(q['disconnected_ids_4_neighbors'],[])
        self.assertEqual(len(q['singleton_ids']),1)
        b2,i2=make_connected_subgroups(p,m)
        np.testing.assert_array_equal(b,b2);self.assertEqual(info,i2)

class PipelineTests(unittest.TestCase):
    def test_candidate_pipeline_and_filename(self):
        import importlib.util
        from unittest.mock import patch
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('mask_loop',root/'scripts/curbd/loop.py')
        loop=importlib.util.module_from_spec(spec);spec.loader.exec_module(loop)
        atlas=np.ones((12,12));atlas[:,6:]=8
        film=np.arange(12*12*8,dtype=float).reshape(8,12,12)
        params={**loop.BASE_PARAMS,'n_pixels':15,'lissage_sigma':0,'segmentation_method':'coherent_v1'}
        with patch.object(loop,'load_dataset',return_value=(film,atlas,np.ones_like(atlas))):
            d=loop.prepare_timeseries((9,6,410),None,params)
        self.assertEqual(d['segmentation_qc']['disconnected_ids_4_neighbors'],[])
        self.assertEqual(d['segmentation_qc']['singleton_ids'],[])
        for unit in range(len(d['ts'])):
            np.testing.assert_allclose(d['ts'][unit],film[:,d['masque_sub']==unit].mean(axis=1))
        self.assertIn('segmentation_methodcoherent_v1',loop.build_run_name(0,(9,6,410),params,[]))
