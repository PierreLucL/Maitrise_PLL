from maitrise_curbd.plotting import plot_curbd_currents_from_pkl, gradient_line
from maitrise_curbd.current_similarity import compare_current_pkls

relative_path = 'Petits n_pixels, gros nRunfree, mais pas 10/run_du_2026-06-25_23h12/config5_M387-6_pix150_sigma3_dffFalse_globalregTrue_nRunTrain1000.pkl'

plot_curbd_currents_from_pkl(
    relative_path,
    gradient_line=gradient_line
)



pkl_a = "/Users/pierre-luclarouche/Desktop/École/Maîtrise/Maitrise_PLL/Moyen n_pixels, Gros nRunTrain/Test n_pixels/config0_M387-6_pix80_sigma3_dffFalse_globalregTrue_nRunTrain1000.pkl"
pkl_b = "/Users/pierre-luclarouche/Desktop/École/Maîtrise/Maitrise_PLL/Petits n_pixels, gros nRunfree, mais pas 10/run_du_2026-06-25_23h12/config1_M387-6_pix80_sigma3_dffFalse_globalregTrue_nRunTrain1000.pkl"

similarity = compare_current_pkls(pkl_a, pkl_b)

pearson = similarity["pearson"]
nrmse = similarity["nrmse"]

print("Pearson 6x6:")
print(pearson)

print("NRMSE 6x6:")
print(nrmse)