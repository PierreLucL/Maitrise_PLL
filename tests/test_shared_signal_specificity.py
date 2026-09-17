"""Vérifications de projection et d'absence de fuite du bloc d'évaluation."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
from shared_signal_specificity import residualize,cca_test

class SharedSignalTests(unittest.TestCase):
    def test_projection_and_no_test_leakage(self):
        rng=np.random.RandomState(4);g=rng.randn(160);fit=np.arange(160)<80
        x=3*g[None,:]+np.array([[2],[5]])
        residual,beta,mean,gm=residualize(x,g,fit)
        np.testing.assert_allclose(residual,0,atol=1e-14)
        y=x.copy();y[:,~fit]+=rng.randn(2,80)*100
        _,b2,m2,g2=residualize(y,g,fit)
        np.testing.assert_array_equal(beta,b2);np.testing.assert_array_equal(mean,m2);self.assertEqual(gm,g2)
    def test_cca_transfer_under_invertible_mixing(self):
        rng=np.random.RandomState(7);x=rng.randn(300,10);y=x@rng.randn(10,10)+4
        fit=np.arange(300)<150;test=~fit
        np.testing.assert_allclose(cca_test(x,y,fit,test),1,atol=1e-12)
    def test_test_sign_is_not_reoriented(self):
        rng=np.random.RandomState(8);x=rng.randn(300,10);y=x.copy();fit=np.arange(300)<150;test=~fit;y[test]*=-1
        np.testing.assert_allclose(cca_test(x,y,fit,test),-1,atol=1e-12)

if __name__=='__main__':unittest.main()
