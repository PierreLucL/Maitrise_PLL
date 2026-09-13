def output_mean_signal_per_regions(folder_path, modality, overwrite=False, regress=False):
    data_path = os.path.join(folder_path, modality + ".tif")
    output_name = modality + "_mean_signals.npy"
    output_path = os.path.join(folder_path, output_name)
    registration_path = os.path.join(folder_path, "atlas.npy")
    mask_path = os.path.join(folder_path, "roi_mask.tif")

    if not os.path.exists(data_path):
        print(f"Data not found at {data_path}. Skipping.")
        return
    elif not os.path.exists(registration_path):
        print(f"Registration not found at {registration_path}. Skipping.")
        return
    elif not os.path.exists(mask_path):
        print(f"Mask not found at {registration_path}. Skipping.")
        return
    elif not overwrite and os.path.exists(output_path):
        print(f"Mean signal already exists at {output_path}. Skipping.")
        return

    # --- Load data ---
    data = tiff.imread(data_path).astype(np.float32)  # shape (T, M, N)
    registration = np.load(registration_path)  # shape (M, N)
    mask = tiff.imread(mask_path)  # shape (M, N), 0 or 255

    # Handle possible 3D mask
    if mask.ndim == 3:
        mask = mask.squeeze()
    mask_bool = mask > 0

    T, M, N = data.shape
    if registration.shape != (M, N):
        raise ValueError(f"Registration shape {registration.shape} does not match data spatial shape {(M, N)}")

    # --- Nuisance signal: mean time course of every pixel OUTSIDE the mask ---
    # Computed on the raw data, before the in-mask pixels are isolated below.
    if regress:
        outside_mask = ~mask_bool
        if np.any(outside_mask):
            outside_signal = np.nanmean(data[:, outside_mask], axis=1).astype(np.float32)  # (T,)
    else:
        outside_signal = np.full(T, np.nan, dtype=np.float32)

    # --- Apply mask ---
    data[:, ~mask_bool] = 0
    data[data == 0] = np.nan

    # --- Compute mean signals for regions 1–12 ---
    region_ids = np.arange(1, 69)
    region_signals_dict = {}
    signals_array = np.full((68, T), np.nan, dtype=np.float32)

    structure = np.ones((3, 3), dtype=bool)
    for i, rid in enumerate(region_ids):
        region_mask = (registration == rid) & mask_bool
        region_mask = binary_erosion(region_mask, structure=structure, iterations=1)
        region_mask = binary_dilation(region_mask, structure=structure, iterations=1)

        if np.any(region_mask):
            region_data = data[:, region_mask]  # shape (T, n_pixels)
            mean_tc = np.nanmean(region_data, axis=1)
            # Regress the outside-mask signal out of this region's time course.
            if regress:
                mean_tc = _regress_out(mean_tc, outside_signal)
        else:
            mean_tc = np.full(T, np.nan, dtype=np.float32)

        region_signals_dict[int(rid)] = mean_tc
        signals_array[i] = mean_tc

    # --- Save output ---
    result = {
        "region_ids": region_ids,
        "signals": signals_array,  # shape (68, T), outside-mask signal regressed out
        "by_region": region_signals_dict,  # dict[int → np.ndarray(T,)]
        "outside_mask_signal": outside_signal,  # (T,) nuisance regressor used
    }

    np.save(output_path, result, allow_pickle=True)
    print(f"Saved mean region signals to {output_path}")

    del data
    del registration
    del mask

    return result
 