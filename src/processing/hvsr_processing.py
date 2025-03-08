import hvsrpy
import matplotlib.pyplot as plt
import numpy as np


def example_site_hvsr():
    f_name = "./data/example_site/453024237.0005.2024.06.09.00.00.00.000.E.miniseed"
    fnames = [[f_name, f_name.replace(".E.", ".N."), f_name.replace(".E.", ".Z.")]]

    srecords, hvsr = microtremor_hvsr_diffuse_field(fnames)

    # Preprocessing Summary
    # preprocessing_settings.psummary()

    # params: window lenth
    # Seismic Recordings - Before Preprocessing (Raw)
    # hvsrpy.plot_seismic_recordings_3c(srecords)
    # Seismic Recordings - After Preprocessing
    # hvsrpy.plot_seismic_recordings_3c(srecords_preprocessed)

    # frequency, amplitude, peak_frequency, peak_amplitude

    #### diffuse field
    # Statistical Summary
    # hvsrpy.summarize_hvsr_statistics(hvsr)
    # hvsrpy.plot_single_panel_hvsr_curves(hvsr)

    # Save results
    # fname_prefix = "example_mhvsr_diffuse_field"
    # fname = f"{fname_prefix}.csv"
    # hvsrpy.object_io.write_hvsr_object_to_file(hvsr, fname)
    # print(f"Results saved successfully to {fname}!")

    return hvsr


#### MICROTREMOR PREPROCESSING
def microtremor_preprocessing(fnames):
    preprocessing_settings = hvsrpy.settings.HvsrPreProcessingSettings()
    preprocessing_settings.detrend = "linear"
    preprocessing_settings.window_length_in_seconds = 100
    preprocessing_settings.orient_to_degrees_from_north = 0.0
    preprocessing_settings.filter_corner_frequencies_in_hz = (None, None)
    preprocessing_settings.ignore_dissimilar_time_step_warning = False

    srecords = hvsrpy.read(fnames)
    srecords_preprocessed = hvsrpy.preprocess(srecords, preprocessing_settings)

    return srecords_preprocessed


#### MICROTREMOR HVSR DIFFUSE FIELD
def microtremor_hvsr_diffuse_field(fnames):
    preprocessing_settings = hvsrpy.settings.HvsrPreProcessingSettings()
    preprocessing_settings.detrend = "linear"
    preprocessing_settings.window_length_in_seconds = 60
    preprocessing_settings.orient_to_degrees_from_north = 0.0
    preprocessing_settings.filter_corner_frequencies_in_hz = (None, None)

    processing_settings = hvsrpy.settings.HvsrDiffuseFieldProcessingSettings()
    processing_settings.window_type_and_width = ("tukey", 0.1)
    processing_settings.smoothing = dict(
        operator="log_rectangular",
        bandwidth=0.1,
        center_frequencies_in_hz=np.geomspace(0.2, 50, 256),
    )

    srecords = hvsrpy.read(fnames)
    srecords = hvsrpy.preprocess(srecords, preprocessing_settings)
    hvsr = hvsrpy.process(srecords, processing_settings)

    return srecords, hvsr
