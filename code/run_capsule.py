""" Writes Units to an NWB file """
import sys
import shutil
import json
from pathlib import Path
import numpy as np
import logging

from uuid import uuid4
import time

import probeinterface as pi
import spikeinterface as si

import h5py
import zarr
from pynwb import NWBHDF5IO
from pynwb.file import Device
from hdmf_zarr import NWBZarrIO

from aind_nwb_utils.utils import get_ephys_devices_from_metadata

from utils import add_waveforms_with_uneven_channels


data_folder = Path("../data")
scratch_folder = Path("../scratch")
results_folder = Path("../results")


def remove_units_table(nwb_path: Path) -> None:
    """Deletes the units table of an NWB file (zarr or HDF5) in place, if there is one."""
    if nwb_path.is_dir():
        if not (nwb_path / "units").is_dir():
            return
        shutil.rmtree(nwb_path / "units")
        # .zmetadata would otherwise still list the deleted group
        zarr.consolidate_metadata(str(nwb_path))
        io_class = NWBZarrIO
    else:
        with h5py.File(nwb_path, "a") as f:
            if "units" not in f:
                return
            # shortcut: HDF5 does not reclaim the space, run h5repack if file size matters
            del f["units"]
        io_class = NWBHDF5IO
    with io_class(str(nwb_path), "r") as io:
        assert io.read().units is None, f"Units table still present in {nwb_path.name}"
    logging.info(f"\tRemoved existing units table from {nwb_path.name}")

# unit properties to skip
skip_unit_properties = [
    "KSLabel",
    "KSLabel_repeat",
    "ContamPct",
    "Amplitude",
]

PARAMS_FILE = Path("../params.json")


def run():
    t_export_start = time.perf_counter()

    if not PARAMS_FILE.is_file():
        raise ValueError(f"Parameters file not found: {PARAMS_FILE}")
    with open(PARAMS_FILE, "r") as f:
        full_params = json.load(f)
    # no "units" section in the pipeline params yet: stub options default to the old CLI defaults
    units_params = full_params["nwb_units"]
    LOGGING = full_params.get("logging", None)

    STUB_TEST = units_params.get("stub", False)
    STUB_UNITS = int(units_params.get("stub_units", 10))

    # find raw data
    ecephys_folders = [
        p
        for p in data_folder.iterdir()
        if p.is_dir()
        and ("ecephys" in p.name or "behavior" in p.name)
        and "sorted" not in p.name and "nwb" not in p.name
        and "ecephys_clipped" not in p.name
    ]
    assert len(ecephys_folders) == 1, "Attach one ecephys folder at a time"
    ecephys_session_folder = ecephys_folders[0]

    if LOGGING is None:
        logging.basicConfig(level="INFO", stream=sys.stdout, format="%(message)s")
    else:
        if LOGGING["package"] == "logging":
            logging_cfg = LOGGING.get("logging_cfg", {})
            logging.basicConfig(stream=sys.stdout, **logging_cfg)
        elif LOGGING["package"] == "log-schema":
            import log_schema

            pipeline_name = LOGGING.get("pipeline_name", "AIND Ephys Pipeline")
            acquisition_name = LOGGING.get("acquisition_name", "unknown")

            config = LOGGING.get("logging_cfg")
            if config is not None and len(config) == 0:
                config = None
            log_schema.setup_logging(
                config=config,
                model={
                    "pipeline_name": pipeline_name,
                    "acquisition_name": acquisition_name,
                    "process_name": "NWB Packaging Units",
                },
            )

    logging.info("Begin processing...", extra={"event_type": "stage_start"})

    logging.info("\n\nNWB EXPORT UNITS")
    logging.info(f"Running NWB conversion with the following parameters:")
    logging.info(f"Stub test: {STUB_TEST}")
    logging.info(f"Stub units: {STUB_UNITS}")

    # find base NWB file
    nwb_files = [
        p
        for p in data_folder.iterdir()
        if (p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")) and "/nwb/" not in str(p)
    ]
    assert len(nwb_files) > 0, "Attach at least one base NWB file"
    nwb_file0 = nwb_files[0]

    if nwb_file0.is_dir():
        assert (nwb_file0 / ".zattrs").is_file(), f"{nwb_file0.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        io_class = NWBZarrIO
    else:
        NWB_BACKEND = "hdf5"
        io_class = NWBHDF5IO
    logging.info(f"NWB backend: {NWB_BACKEND}")

    # start by copying NWB files to results folder, since we will write in append mode
    # and we want to avoid modifying input files in place
    for nwb_file_path in nwb_files:
        if NWB_BACKEND == "zarr":
            shutil.copytree(nwb_file_path, results_folder / nwb_file_path.name)
        else:  # HDF5
            shutil.copyfile(nwb_file_path, results_folder / nwb_file_path.name)

    # find raw data
    job_json_files = [p for p in data_folder.glob('**/*.json') if "job" in p.name]
    job_dicts = []
    recording_names_in_json = []
    for job_json_file in job_json_files:
        with open(job_json_file) as f:
            job_dict = json.load(f)
            recording_names_in_json.append(job_dict["recording_name"])
        job_dicts.append(job_dict)
    logging.info(f"Found {len(job_dicts)} JSON job files")

    # check for timestamps to overwrite recording timestamps
    timestamps_folder = data_folder / "timestamps"

    # find sorted data
    sorted_folders = [
        p for p in data_folder.iterdir() if p.is_dir() and "sorted" in p.name and "spikesorted" not in p.name
    ]

    if len(sorted_folders) == 0:
        # pipeline mode
        if (data_folder / "collect_pipeline_output_test").is_dir():
            logging.info("\n*******************\n**** TEST MODE ****\n*******************\n")
            sorted_folder = data_folder / "collect_pipeline_output_test"
        else:
            sorted_folder = data_folder
    elif len(sorted_folders) == 1:
        # capsule mode
        sorted_folder = sorted_folders[0]

    postprocessed_folder = sorted_folder / "postprocessed"
    spikesorted_folder = sorted_folder / "spikesorted"
    if not postprocessed_folder.is_dir():
        logging.info("Postprocessed folder not found. Skipping NWB export")
        # create dummy nwb folder to avoid pipeline failure
        error_txt = results_folder / "error.txt"
        error_txt.write_text("Postprocessed folder not found. No NWB files were created.")
    else:
        # we create a result NWB file for each experiment/recording# we create a result NWB file for each experiment/recording
        postprocessed_folders = sorted([p.name for p in postprocessed_folder.iterdir() if p.is_dir()])
        recording_names = [p[:p.find(".zarr")] if p.endswith(".zarr") else p for p in postprocessed_folders]
        logging.info(f"Found {len(recording_names)} processed recordings")

        # find blocks and recordings
        block_ids = []
        recording_ids = []
        group_ids = []
        stream_names = []

        # these dictionaries are used to check if recordings from the same block, but from different streams
        # or groups have the same sampling frequency and to find the number of sparse channels per unit:
        # waveform_means and standard_deviations require the same shape across all units
        recording_sampling_frequencies = {}
        # (block, recording) -> list of (max sparse channels per unit, num analyzer channels), one per stream/group
        analyzer_num_channels = {}
        write_waveforms = {}
        # (block, recording) pairs that actually exist: not all streams/recordings are present in every block
        existing_block_recordings = set()
        for recording_name in recording_names:
            if "group" in recording_name:
                block_str = recording_name.split("_")[0]
                recording_str = recording_name.split("_")[-2]
                group_str = recording_name.split("_")[-1]
                stream_name = "_".join(recording_name.split("_")[1:-2])
            else:
                block_str = recording_name.split("_")[0]
                recording_str = recording_name.split("_")[-1]
                stream_name = "_".join(recording_name.split("_")[1:-1])
                group_str = ""

            existing_block_recordings.add((block_str, recording_str))
            if block_str not in block_ids:
                block_ids.append(block_str)
            if recording_str not in recording_ids:
                recording_ids.append(recording_str)
            if stream_name not in stream_names:
                stream_names.append(stream_name)
            if group_str not in group_ids:
                group_ids.append(group_str)

            # load the recording and check sampling rate and number of sparse channels
            recording_job_dict = None
            for job_dict in job_dicts:
                if recording_name == job_dict["recording_name"]:
                    recording_job_dict = job_dict
                    break
            if recording_job_dict is None:
                logging.info(f"Couldn't find job dict for {recording_name}")
                continue
            analyzer_folder = postprocessed_folder / f"{recording_name}.zarr"
            if not analyzer_folder.is_dir():
                analyzer_folder = postprocessed_folder / recording_name
            if not analyzer_folder.is_dir():
                logging.info(f"No analyzer found for {recording_name}")
                continue
            key = (block_str, recording_str)
            recording = si.load(recording_job_dict["recording_dict"], base_folder=data_folder)
            recording_sampling_frequencies.setdefault(key, {})[stream_name] = recording.sampling_frequency
            analyzer_lazy = si.load(analyzer_folder, lazy=True)
            num_analyzer_channels = analyzer_lazy.get_num_channels()
            if analyzer_lazy.sparsity is None:
                num_sparse_channels = num_analyzer_channels
            else:
                num_sparse_channels = int(analyzer_lazy.sparsity.mask.sum(axis=1).max())
            analyzer_num_channels.setdefault(key, {})[recording_name] = (num_sparse_channels, num_analyzer_channels)

        # We first check the sampling frequencies across streams.
        # If sampling frequencies for different streams, do not write waveforms for the block/recording, because they
        # have a different number of samples
        for key, sampling_frequencies in recording_sampling_frequencies.items():
            if len(np.unique(list(sampling_frequencies.values()))) > 1:
                logging.info(
                    f"Recordings from different blocks/groups have different sampling frequencies: {recording_sampling_frequencies}"
                )
                write_waveforms[key] = False
            else:
                write_waveforms[key] = True

        # Templates are written sparse: every unit gets the same number of channels, i.e. the maximum
        # number of sparse channels across streams/groups of the block/recording.
        # If a stream/group has fewer channels than that (e.g., a 32-channel probe next to a 384-channel probe),
        # we cannot write waveforms.
        num_channels_max = {}
        for key, num_channels in analyzer_num_channels.items():
            num_channels_max[key] = max(n_sparse for n_sparse, _ in num_channels.values())
            logging.info(f"Sparse templates for {key}: writing {num_channels_max[key]} channels per unit")
            for name, (n_sparse, n_total) in num_channels.items():
                logging.info(f"\t{name}: max sparse channels {n_sparse}, total channels {n_total}")
            if num_channels_max[key] > min(n_total for _, n_total in num_channels.values()):
                logging.info(
                    f"Max number of sparse channels ({num_channels_max[key]}) exceeds the number of channels of "
                    f"some streams/groups for {key}: {num_channels}. Skipping waveforms."
                )
                write_waveforms[key] = False

        block_ids = sorted(block_ids)
        recording_ids = sorted(recording_ids)
        streams_to_process = sorted(stream_names)

        logging.info(f"Number of NWB files to write: {len(existing_block_recordings)}")
        logging.info(f"Number of streams to write for each file: {len(streams_to_process)}")

        nwb_output_files = []
        multi_input_files = False
        if len(nwb_files) > 1:
            assert len(nwb_files) >= len(existing_block_recordings), (
                "Inconsistent number of input NWB files with number of blocks and recordings: "
                f"Num NWB files: {len(nwb_files)} - Num files to write: {len(existing_block_recordings)}"
            )
            multi_input_files = True

        for block_index, block_str in enumerate(block_ids):
            for segment_index, recording_str in enumerate(recording_ids):
                if (block_str, recording_str) not in existing_block_recordings:
                    logging.info(f"No recordings found for {block_str} - {recording_str}. Skipping")
                    continue
                # add recording/experiment id if needed
                if multi_input_files:
                    nwb_input_path_for_current = [
                        p for p in nwb_files if f"{block_str}_" in p.stem and
                        (p.stem.endswith(recording_str) or p.stem.endswith(f"{recording_str}_stub"))
                    ]
                    assert len(nwb_input_path_for_current) == 1, (
                        f"Could not find input NWB file for {block_str}-{recording_str}. Available NWB files are: {nwb_files}"
                    )
                    nwbfile_input_path = nwb_input_path_for_current[0]
                    logging.info(f"Found input NWB file for {block_str}-{recording_str}: {nwbfile_input_path.name}")
                    nwbfile_output_path = results_folder / f"{nwbfile_input_path.stem}.nwb"
                    # in this case the nwb files have been already copied to the results folder
                else:
                    nwbfile_input_path = nwb_files[0]
                    nwb_original_file_name = nwbfile_input_path.stem
                    if block_str in nwb_original_file_name and recording_str in nwb_original_file_name:
                        nwb_file_name = f"{nwb_original_file_name}.nwb"
                    else:
                        nwb_file_name = f"{nwb_original_file_name}_{block_str}_{recording_str}.nwb"
                    nwbfile_output_path = results_folder / nwb_file_name

                    # copy nwb input file to results to read in append mode
                    if NWB_BACKEND == "zarr":
                        if not nwbfile_output_path.is_dir():
                            shutil.copytree(nwbfile_input_path, nwbfile_output_path)
                    else:  # HDF5
                        if not nwbfile_output_path.is_file():
                            shutil.copyfile(nwbfile_input_path, nwbfile_output_path)

                # Find probe devices (this will only work for AIND)
                devices_from_metadata, target_locations = get_ephys_devices_from_metadata(
                    ecephys_session_folder
                )

                # an input NWB from a previous run (e.g. curation pipeline) has a units table to replace
                remove_units_table(nwbfile_output_path)

                with io_class(str(nwbfile_output_path), "a") as append_io:
                    nwbfile = append_io.read()

                    added_stream_names = []
                    for stream_name in streams_to_process:
                        stream_str = str(stream_name)
                        for group_str in group_ids:
                            recording_name = f"{block_str}_{stream_name}_{recording_str}"
                            if group_str != "":
                                recording_name += f"_{group_str}"
                                stream_str += f"_{group_str}"
                            if not (postprocessed_folder / f"{recording_name}.zarr").is_dir():
                                logging.info(f"Postprocessed folder not found for {recording_name}")
                                continue

                            # load JSON and recordings
                            recording_job_dict = None
                            for job_dict in job_dicts:
                                if job_dict["recording_name"] == recording_name:
                                    recording_job_dict = job_dict
                                    break
                            if recording_job_dict is None:
                                logging.info(f"Could not find JSON file associated to {recording_name}")
                                continue

                            added_stream_names.append(stream_str)

                            # load associated recordings
                            recording = si.load(recording_job_dict["recording_dict"], base_folder=data_folder)
                            skip_times = job_dict.get("skip_times", False)
                            if skip_times:
                                recording.reset_times()
                            timestamps_file = timestamps_folder / f"{recording_name}.npy"
                            if timestamps_file.is_file():
                                logging.info(f"\tSetting synced timestamps from {timestamps_file}")
                                timestamps = np.load(timestamps_file)
                                recording.set_times(timestamps)
                            # a zarr-backed time vector stays lazy, and neuroconv indexes it once per unit,
                            # decoding the whole vector each time; a full-range get_times caches it in memory
                            for segment_index in range(recording.get_num_segments()):
                                if recording.has_time_vector(segment_index):
                                    recording.get_times(segment_index)

                            # Add device and electrode group
                            probegroup = recording.get_probegroup()
                            assert len(probegroup.probes) == 1, (
                                "Grouping failed for this session. Each stream should be associated with a single probe!"
                            )
                            probe = probegroup.probes[0]
                            electrode_group_location = "unknown"

                            # 1. Look for AIND devices in metadata and use them if they match the stream name
                            probe_device_name = None
                            if devices_from_metadata:
                                for device_name, device in devices_from_metadata.items():
                                    # add the device, since it could be a laser
                                    if device_name not in nwbfile.devices:
                                        nwbfile.add_device(device)
                                    # find probe device name
                                    probe_no_spaces = device_name.replace(" ", "")
                                    if probe_no_spaces in stream_name:
                                        probe_device_name = device_name
                                        electrode_group_location = target_locations.get(device_name, "unknown")
                                        logging.info(
                                            f"\tFound device from metadata: {probe_device_name} at location {electrode_group_location}"
                                        )
                                        # 1a. Apply fix for Quad Base probes to get probe device name from probe metadata instead of rig.json,
                                        # since rig.json has the same name for all shanks but we need to differentiate them
                                        model_name = probe.model_name
                                        model_description = probe.description
                                        if (model_name is not None and "Quad Base" in model_name) or \
                                            (model_description is not None and "Quad Base" in model_description):
                                            logging.info(f"Detected Quad Base: changing name from {probe_device_name} to {probe.name}")
                                            probe_device_name = probe.name
                                        break

                            # 2. If no metadata devices, use probeinterface probes info from recording annotations
                            if probe_device_name is None:
                                probe_device_name = probe.name or probe.model_name or "Probe"
                                logging.info(f"\tAdding probe information from recording metadata")

                            # 3. Add probe to NWB
                            probe_device_manufacturer = probe.manufacturer
                            probe_model_name = probe.model_name
                            probe_serial_number = probe.serial_number
                            probe_description = probe.description
                            probe_device_description = ""

                            if probe_model_name is not None:
                                probe_device_description += f"Model: {probe_model_name}"
                            if probe_serial_number is not None:
                                if len(probe_device_description) > 0:
                                    probe_device_description += " - "
                                probe_device_description += f"Serial number: {probe_serial_number}"
                            if probe_description is not None:
                                if len(probe_device_description) > 0:
                                    probe_device_description += " - "
                                probe_device_description += f"Description: {probe_description}"
                            probe_device = Device(
                                name=probe_device_name,
                                description=probe_device_description,
                                manufacturer=probe_device_manufacturer,
                            )
                            if probe_device_name not in nwbfile.devices:
                                nwbfile.add_device(probe_device)
                                logging.info(f"\tAdded probe device: {probe_device.name} - {probe_device.description}")

                            # Add electrode metadats
                            electrode_metadata = dict(
                                Ecephys=dict(
                                    Device=[dict(name=probe_device_name)],
                                )
                            )

                            if (postprocessed_folder / f"{recording_name}.zarr").is_dir():
                                # zarr format
                                analyzer_folder = postprocessed_folder / f"{recording_name}.zarr"
                            elif (postprocessed_folder / recording_name).is_dir():
                                # binary format
                                analyzer_folder = postprocessed_folder / recording_name
                            else:
                                logging.info(f"No analyzer found for {recording_name}")
                                continue

                            analyzer = si.load(analyzer_folder, load_extensions=False)
                            sorting_curated = analyzer.sorting

                            if len(analyzer.unit_ids) != len(np.unique(analyzer.unit_ids)):
                                try:
                                    analyzer.sorting = analyzer.sorting.rename_units(sorting_curated.unit_ids)
                                    logging.info(
                                        f"Wrong unit ids for analyzer for {recording_name}. "
                                        "Resetting unit ids with curated sorting"
                                    )
                                except Exception as e:
                                    logging.info(
                                        f"Wrong unit ids and resetting units failed. Skipping {recording_name}"
                                    )
                                    continue

                            # Add unit properties (UUID and probe info, ks_unit_id)
                            unit_uuids = [str(uuid4()) for u in sorting_curated.unit_ids]
                            sorting_curated.set_property(
                                "device_name", [probe_device_name] * sorting_curated.get_num_units()
                            )
                            sorting_curated.set_property("shank", [group_str] * sorting_curated.get_num_units())
                            sorting_curated.set_property("unit_name", unit_uuids)
                            sorting_curated.set_property("ks_unit_id", sorting_curated.unit_ids)

                            # Add 'amplitude' property
                            amplitudes = si.get_template_amplitude_on_main_channel(analyzer, peak_mode="peak_to_peak", with_dict=False)
                            sorting_curated.set_property("amplitude", amplitudes)
                            # Add depth property
                            unit_locations = np.round(analyzer.get_extension("unit_locations").get_data(), 2)
                            sorting_curated.set_property("estimated_x", unit_locations[:, 0])
                            sorting_curated.set_property("estimated_y", unit_locations[:, 1])
                            if unit_locations.shape[1] == 3:
                                sorting_curated.set_property("estimated_z", unit_locations[:, 2])
                            sorting_curated.set_property("depth", unit_locations[:, 1])
                            # add max_channel property
                            extremum_channel_indices = list(si.get_template_extremum_channel(analyzer, outputs="index").values())
                            sorting_curated.set_property("extremum_channel_index", extremum_channel_indices)

                            if STUB_TEST:
                                logging.info(f"\tSelecting {STUB_UNITS} units for testing")
                                stub_unit_ids = sorting_curated.unit_ids[:STUB_UNITS]
                                sorting_curated = sorting_curated.select_units(stub_unit_ids)
                                analyzer = analyzer.select_units(stub_unit_ids)

                            logging.info(f"\tAdding {len(sorting_curated.unit_ids)} units from {recording_name}")

                            # Register recording for precise timestamps
                            sorting_curated.register_recording(recording)
                            analyzer.sorting = sorting_curated

                            # Retrieve sorter name
                            sorter_log_file = spikesorted_folder / recording_name / "spikeinterface_log.json"
                            units_description = "Units"
                            if sorter_log_file.is_file():
                                with open(sorter_log_file, "r") as f:
                                    sorter_log = json.load(f)
                                    sorter_name = sorter_log["sorter_name"]
                                    units_description += f" from {sorter_name.capitalize()}"

                            channel_groups = recording.get_channel_groups()
                            if group_str == "":
                                # single shank probe
                                recording.set_channel_groups([probe_device_name] * recording.get_num_channels())
                                electrode_groups_metadata = [
                                    dict(
                                        name=probe_device_name,
                                        description=f"Recorded electrodes from probe {probe_device_name}",
                                        location=electrode_group_location,
                                        device=probe_device_name,
                                    )
                                ]
                            else:
                                recording.set_channel_groups([f"{probe_device_name}_group{g}" for g in channel_groups])
                                channel_groups_unique = np.unique(recording.get_channel_groups())
                                electrode_groups_metadata = [
                                    dict(
                                        name=group,
                                        description=f"Recorded electrodes from group {group}",
                                        location=electrode_group_location,
                                        device=probe_device_name,
                                    )
                                    for group in channel_groups_unique
                                ]
                            electrode_metadata["Ecephys"]["ElectrodeGroup"] = electrode_groups_metadata

                            add_waveforms_with_uneven_channels(
                                sorting_analyzer=analyzer,
                                recording=recording,
                                nwbfile=nwbfile,
                                metadata=electrode_metadata,
                                skip_properties=skip_unit_properties,
                                units_description=units_description,
                                write_waveforms=write_waveforms[(block_str, recording_str)],
                                num_channels_max=num_channels_max.get((block_str, recording_str)),
                            )
                    logging.info(f"Added {len(added_stream_names)} streams")

                    if NWB_BACKEND == "zarr":
                        write_args = {'link_data': False}
                    else:
                        write_args = {}

                    t_write_start = time.perf_counter()
                    append_io.write(nwbfile)
                    t_write_end = time.perf_counter()
                    elapsed_time_write = np.round(t_write_end - t_write_start, 2)
                    logging.info(f"Writing time: {elapsed_time_write}s")
                    # with io_class(str(nwbfile_output_path), "w") as export_io:
                    #    export_io.export(src_io=read_io, nwbfile=nwbfile, write_args=write_args)
                    logging.info(f"Done writing {nwbfile_output_path}")
                    nwb_output_files.append(nwbfile_output_path)

    t_export_end = time.perf_counter()
    elapsed_time_export = np.round(t_export_end - t_export_start, 2)
    logging.info(f"NWB EXPORT UNITS time: {elapsed_time_export}s")
    logging.info("Pipeline stage completed", extra={"event_type": "stage_complete"})


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        logging.exception("Pipeline stage failed", extra={"event_type": "stage_error"})
        raise
