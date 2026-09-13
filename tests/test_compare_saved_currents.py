"""Cas scientifiques minimaux pour la comparaison des courants sauvegardes."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

path=Path(__file__).resolve().parents[1]/'scripts/analysis/compare_saved_currents.py'
spec=importlib.util.spec_from_file_location('compare_saved_currents',path)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CurrentComparisonTests(unittest.TestCase):
    def setUp(self):
        self.time=np.arange(10,dtype=float)/12
        self.names=np.array(['A','B'])
        self.labels=np.array([[0,0],[0,1],[1,0],[1,1]])
        self.cur=np.random.default_rng(14).normal(size=(4,10))

    def data(self,cur=None,t=None,labels=None,names=None):
        return m.validate_currents(self.cur if cur is None else cur,
                                   self.labels if labels is None else labels,
                                   self.time if t is None else t,
                                   self.names if names is None else names)

    def test_identical_and_offset_scale(self):
        r=m.compare_currents(self.data(),self.data(cur=2*self.cur+4))
        self.assertAlmostEqual(r['median_pearson'],1)
        for p in r['pairs']:
            self.assertAlmostEqual(p['std_ratio_b_over_a'],2)
            self.assertAlmostEqual(p['mean_b'],2*p['mean_a']+4)

    def test_anatomical_and_row_reordering(self):
        order=np.array([2,0,3,1])
        b=self.data(cur=self.cur[order], labels=(1-self.labels)[order],names=self.names[::-1])
        self.assertAlmostEqual(m.compare_currents(self.data(),b)['median_pearson'],1)

    def test_times_must_match_even_at_same_length(self):
        with self.assertRaises(ValueError):
            m.compare_currents(self.data(),self.data(t=self.time+.1))

    def test_unequal_lengths_rejected(self):
        with self.assertRaises(ValueError):
            m.compare_currents(self.data(),self.data(cur=self.cur[:,:-1],t=self.time[:-1]))

    def test_invalid_labels_names_and_values(self):
        for kwargs in [dict(labels=np.zeros((4,2),int)),dict(labels=self.labels.astype(float)),
                       dict(names=np.array(['A','A'])),dict(t=self.time[::-1]),
                       dict(cur=np.full((4,10),np.nan))]:
            with self.subTest(kwargs=list(kwargs)),self.assertRaises(ValueError):
                self.data(**kwargs)
        with self.assertRaises(ValueError):
            m.compare_currents(self.data(),self.data(names=np.array(['A','C'])))

    def test_constant_trace_is_undefined(self):
        r=m.compare_currents(self.data(cur=np.ones((4,10))),self.data())
        self.assertEqual(r['n_valid_pearson'],0)
        self.assertIsNone(r['median_pearson'])

    def test_compensating_currents_leave_total_unchanged(self):
        changed=self.cur.copy()
        changed[0]+=10*self.cur[2]
        changed[1]-=10*self.cur[2]
        r=m.compare_currents(self.data(),self.data(cur=changed))
        self.assertAlmostEqual(r['total_recurrent_by_target'][0]['pearson'],1)
        self.assertLess(r['pairs'][0]['pearson'],.95)

    def test_npz_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'currents.npz'
            np.savez(path,currents=self.cur,labels=self.labels,tRNN=self.time,region_names=self.names.astype(object))
            self.assertAlmostEqual(m.compare_currents(self.data(),m.load_currents(path))['median_pearson'],1)


if __name__=='__main__':
    unittest.main()
