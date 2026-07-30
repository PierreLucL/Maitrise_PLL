from maitrise_curbd.plotting import plot_curbd_currents_from_pkl, gradient_line, plot_current_similarity_imshow
from maitrise_curbd.current_similarity import compare_current_pkls
import matplotlib.pyplot as plt
import pickle


pkl_a = "/Users/pierre-luclarouche/Desktop/École/Maîtrise/Maitrise_PLL/NEW DATASETS À SOIR/run_du_2026-07-30_02h25/config0_191_pix50_sigma2_dffFalse_globalregTrue_nRunTrain1000.pkl"
#pkl_b = "/Users/pierre-luclarouche/Desktop/École/Maîtrise/Maitrise_PLL/Petits n_pixels, gros nRunfree, mais pas 10/run_du_2026-06-25_23h12/config8_M396-6_pix80_sigma2_dffFalse_globalregTrue_nRunTrain1000.pkl"

plot_curbd_currents_from_pkl(
    pkl_a,
    gradient_line=gradient_line, sigma=1
)

#plot_curbd_currents_from_pkl(
#    pkl_b,
#    gradient_line=gradient_line, sigma=1
#)

def get_title_info_from_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    souris = data["row"]["souris"]
    n_pixels = data["row"]["n_pixels"]

    return souris, n_pixels

#similarity = compare_current_pkls(pkl_a, pkl_b)

souris_a, n_pixels_a = get_title_info_from_pkl(pkl_a)
#souris_b, n_pixels_b = get_title_info_from_pkl(pkl_b)

title = (
    "Pearson des courants CURBD\n"
    f"{souris_a}, n_pixels {n_pixels_a} vs {souris_b}, n_pixels {n_pixels_b}"
)

#pearson = similarity["pearson"]
#nrmse = similarity["nrmse"]

#fig, ax = plot_current_similarity_imshow(
#    nrmse,
#    title=title,
#)

plt.show()