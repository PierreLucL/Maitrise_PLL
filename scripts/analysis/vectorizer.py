### Mini-script de vectorisation post-run.
### Tu pointes vers un .pkl CURBD, ca sort les courants en metriques compactes.

from maitrise_curbd.plotting import plot_curbd_currents_from_pkl, gradient_line, plot_current_similarity_imshow
from maitrise_curbd.current_similarity import compare_current_pkls
import matplotlib.pyplot as plt
from maitrise_curbd.vector import load_curbd_currents_array, compute_curbd_metrics, plot_curbd_metrics
import pickle

### Fichier source a analyser. A remplacer par un pkl dans results/ quand tu lances pour vrai.
pkl_a = "/Users/pierre-luclarouche/Desktop/École/Maîtrise/Maitrise_PLL/Results/NEW DATASETS À SOIR/run_du_2026-07-30_02h25/config1_191_pix80_sigma2_dffFalse_globalregTrue_nRunTrain1000.pkl"

### On en tire les courants CURBD dans un array de taille (n_regions^2, T).
currents = load_curbd_currents_array(pkl_a)

### On calcule les 5 metriques par courant: RMS, P95-P5, AUC, skewness, kurtosis.
metrics = compute_curbd_metrics(
    currents,
    dt=1/12,  # pas de temps de données (en secondes)
)

### Petit plot final pour voir vite quels courants ressortent.
plot_curbd_metrics(metrics)
