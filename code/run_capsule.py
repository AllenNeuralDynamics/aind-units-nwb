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

# needed to lead extensions
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm

from pynwb import NWBHDF5IO
from pynwb.file import Device
from hdmf_zarr import NWBZarrIO

# AIND
try:
    from aind_log_utils import log

    HAVE_AIND_LOG_UTILS = True
except ImportError:
    HAVE_AIND_LOG_UTILS = False

from utils import get_devices_from_rig_metadata, add_waveforms_with_uneven_channels


data_folder = Path("../data")
scratch_folder = Path("../scratch")
results_folder = Path("../results")

# unit properties to skip
skip_unit_properties = [
    "KSLabel",
    "KSLabel_repeat",
    "ContamPct",
    "Amplitude",
]


if __name__ == "__main__":
    t_export_start = time.perf_counter()

    # find raw data
    ecephys_folders = [
        p
        for p in data_folder.iterdir()
        if p.is_dir()
        and ("ecephys" in p.name or "behavior" in p.name)
        and "sorted" not in p.name and "nwb" not in p.name
    ]
    assert len(ecephys_folders) == 1, "Attach one ecephys folder at a time"
    ecephys_session_folder = ecephys_folders[0]
    if HAVE_AIND_LOG_UTILS:
        # look for subject.json and data_description.json files
        subject_json = ecephys_session_folder / "subject.json"
        subject_id = "undefined"
        if subject_json.is_file():
            subject_data = json.load(open(subject_json, "r"))
            subject_id = subject_data["subject_id"]

        data_description_json = ecephys_session_folder / "data_description.json"
        session_name = "undefined"
        if data_description_json.is_file():
            data_description = json.load(open(data_description_json, "r"))
            session_name = data_description["name"]

        log.setup_logging(
            "NWB Packaging Units",
            subject_id=subject_id,
            asset_name=session_name,
        )
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    logging.info("\n\nNWB EXPORT UNITS")

    # find base NWB file
    nwb_files = [
        p
        for p in data_folder.iterdir()
        if (p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")) and "/nwb/" not in str(p)
    ]
    assert len(nwb_files) > 0, "Attach at least one base NWB file"
    nwbfile_input_path = nwb_files[0]

    if nwbfile_input_path.is_dir():
        assert (nwbfile_input_path / ".zattrs").is_file(), f"{nwbfile_input_path.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        io_class = NWBZarrIO
    else:
        NWB_BACKEND = "hdf5"
        io_class = NWBHDF5IO
    logging.info(f"NWB backend: {NWB_BACKEND}")

    # if more than 1 input NWB files, we copy them all to the results
    # since some processing might have failed
    if len(nwb_files) > 1:
        for nwb_file_path in nwb_files:
            if nwb_file_path.is_dir():
                shutil.copytree(nwb_file_path, results_folder / nwb_file_path.name)
            else:
                shutil.copyfile(nwb_file_path, results_folder / nwb_file_path.name)

    # find raw data
    job_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
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
    curated_folder = sorted_folder / "curated"
    spikesorted_folder = sorted_folder / "spikesorted"
    if not postprocessed_folder.is_dir():
        logging.info("Postprocessed folder not found. Skipping NWB export")
        # create dummy nwb folder to avoid pipeline failure
        error_txt = results_folder / "error.txt"
        error_txt.write_text("Postprocessed folder not found. No NWB files were created.")
    else:
        assert curated_folder.is_dir(), f"Curated folder {curated_folder} does not exist"
        assert spikesorted_folder.is_dir(), f"Spikesorted folder {spikesorted_folder} does not exist"

        # we create a result NWB file for each experiment/recording
        recording_names = sorted([p.name for p in curated_folder.iterdir() if p.is_dir()])
        logging.info(f"Found {len(recording_names)} processed recordings")

        # find blocks and recordings
        block_ids = []
        recording_ids = []
        group_ids = []
        stream_names = []

        # these dictionaries are used to check if recordings from the same block, but from different streams
        # or groups have the same sampling frequency and number of channels
        # if not, we skip writing waveform_means and standard_deviations since they require the same
        # shape across all units
        recording_sampling_frequencies = {}
        recording_num_channels = {}
        recording_num_channels_all = {}
        write_waveforms = {}
        aggregate_groups = {}
        # since we load recordings here, we keep a list of recordings associated to each recording name
        # as we might need it when it comes to aggregating groups
        recordings_associated_to_recording_name = {}
        for recording_name in recording_names:
            if "group" in recording_name:
                block_str = recording_name.split("_")[0]
                recording_str = recording_name.split("_")[-2]
                group_str = recording_name.split("_")[-1]
                stream_name = "_".join(recording_name.split("_")[1:-2])
                recording_name_no_group = "_".join(recording_name.split("_")[:-1])
            else:
                block_str = recording_name.split("_")[0]
                recording_str = recording_name.split("_")[-1]
                stream_name = "_".join(recording_name.split("_")[1:-1])
                group_str = ""
                recording_name_no_group = recording_name

            if block_str not in block_ids:
                block_ids.append(block_str)
            if recording_str not in recording_ids:
                recording_ids.append(recording_str)
            if stream_name not in stream_names:
                stream_names.append(stream_name)
            if group_str not in group_ids:
                group_ids.append(group_str)

            # load the recording and check sampling rate and number of channels
            recording_job_dict = None
            recording_job_dicts_all = []
            for job_dict in job_dicts:
                if recording_name_no_group in job_dict["recording_name"]:
                    recording_job_dicts_all.append(job_dict)
                if recording_name == job_dict["recording_name"]:
                    recording_job_dict = job_dict
            if len(recording_job_dicts_all) > 0 and recording_job_dict is not None:
                recording = si.load(recording_job_dict["recording_dict"], base_folder=data_folder)
                recording_list = [si.load(job_dict["recording_dict"], base_folder=data_folder) for job_dict in recording_job_dicts_all]
                recordings_associated_to_recording_name[recording_name] = recording_list
                if (block_str, recording_str) not in recording_sampling_frequencies:
                    recording_sampling_frequencies[(block_str, recording_str)] = {}
                if (block_str, recording_str) not in recording_num_channels:
                    recording_num_channels[(block_str, recording_str)] = {}
                if (block_str, recording_str) not in recording_num_channels_all:
                    recording_num_channels_all[(block_str, recording_str)] = {}
                recording_sampling_frequencies[(block_str, recording_str)][stream_name] = recording.sampling_frequency
                recording_num_channels[(block_str, recording_str)][(stream_name, group_str)] = recording.get_num_channels()
                recording_num_channels_all[(block_str, recording_str)][stream_name] = sum([r.get_num_channels() for r in recording_list])
            else:
                logging.info(f"Couldn't find job dict for {recording_name}")

        # We first check the sampling frequencies across streams.
        # If sampling frequencies for different streams, do not write waveforms for the block/recording, because they
        # have a different number of samples
        for key, sampling_frequencies in recording_sampling_frequencies.items():
            if len(np.unique(sampling_frequencies.values())) > 1:
                logging.info(
                    f"Recordings from different blocks/groups have different sampling frequencies: {recording_sampling_frequencies}"
                )
                write_waveforms[key] = False
            else:
                write_waveforms[key] = True
        # Here we check the number of channels across streams, we have 3 options:
        # 1. there are no channel groups OR there are channel groups with same number of channels for each stream
        #    (.e.g, 2 NP2.0-4shank in the same experiment/recording --> each group has 96 electrodes)
        # 2. there are channel groups, but in the same experiment/recording there is a mix of probes with/without groups,
        #    but the sum of channels per probe is the same
        #    (e.g. 1 NP2.0-4shank and 1 NP1.0 --> each shank has 96 electrodes, but the second probe has 384
        # 3. there are channel groups/probes with incompatible number of channels.
        #    (e.g., a 32-channel probe and a 384-channel probe in the same experiment/recording)
        # For case 1, we can write waverorm_means/stds for individual groups only (96-channels)
        # For case 2, we need to aggregate groups and pad waveforms with zeros
        # For case 3, we cannot write waveforms
        for key, num_channels in recording_num_channels.items():
            if write_waveforms[key]:
                if len(np.unique(num_channels.values())) == 1:
                    write_waveforms[key] = True
                    aggregate_groups[key] = False
                else:
                    for key_all, num_channels_all in recording_num_channels_all.items():
                        if len(np.unique(num_channels_all.values())) == 1:
                            write_waveforms[key] = True
                            aggregate_groups[key] = True
                        else:
                            logging.info(
                                f"Recordings from different blocks/groups have different number of channels: {recording_num_channels}"
                            )
                            write_waveforms[key] = False
                            aggregate_groups[key] = False

        block_ids = sorted(block_ids)
        recording_ids = sorted(recording_ids)
        stream_names = sorted(stream_names)

        logging.info(f"Number of NWB files to write: {len(block_ids) * len(recording_ids)}")
        logging.info(f"Number of streams to write for each file: {len(stream_names)}")

        nwb_output_files = []
        multi_input_files = False
        if len(nwb_files) > 1:
            assert len(nwb_files) >= len(block_ids) * len(recording_ids), (
                "Inconsistent number of input NWB files with number of blocks and recordings: "
                f"Num NWB files: {len(nwb_files)} - Num files to write: {len(block_ids) * len(recording_ids)}"
            )
            multi_input_files = True

        for block_index, block_str in enumerate(block_ids):
            for segment_index, recording_str in enumerate(recording_ids):
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
                    nwb_original_file_name = nwbfile_input_path.stem
                    if block_str in nwb_original_file_name and recording_str in nwb_original_file_name:
                        nwb_file_name = f"{nwb_original_file_name}.nwb"
                    else:
                        nwb_file_name = f"{nwb_original_file_name}_{block_str}_{recording_str}.nwb"
                    nwbfile_output_path = results_folder / nwb_file_name

                    # copy nwb input file to results to read in append mode
                    if nwbfile_input_path.is_dir():
                        shutil.copytree(nwbfile_input_path, nwbfile_output_path)
                    else:
                        shutil.copyfile(nwbfile_input_path, nwbfile_output_path)

                # Find probe devices (this will only work for AIND)
                devices_from_rig, target_locations = get_devices_from_rig_metadata(
                    ecephys_session_folder, segment_index=segment_index
                )

                with io_class(str(nwbfile_output_path), "a") as append_io:
                    nwbfile = append_io.read()

                    added_stream_names = []
                    for stream_name in stream_names:
                        stream_str = str(stream_name)
                        for group_str in group_ids:
                            recording_name = f"{block_str}_{stream_name}_{recording_str}"
                            if group_str != "":
                                recording_name += f"_{group_str}"
                                stream_str += f"_{group_str}"
                            if not (curated_folder / recording_name).is_dir():
                                logging.info(f"Curated units for {recording_name} not found.")
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
                            recording = si.load(job_dict["recording_dict"], base_folder=data_folder)
                            skip_times = job_dict.get("skip_times", False)
                            if skip_times:
                                recording.reset_times()
                            timestamps_file = timestamps_folder / f"{recording_name}.npy"
                            if timestamps_file.is_file():
                                logging.info(f"\tSetting synced timestamps from {timestamps_file}")
                                timestamps = np.load(timestamps_file)
                                recording.set_times(timestamps)

                            # Add device and electrode group
                            probe_device_name = None
                            if devices_from_rig:
                                for device_name, device in devices_from_rig.items():
                                    # add the device, since it could be a laser
                                    if device_name not in nwbfile.devices:
                                        nwbfile.add_device(device)
                                    # find probe device name
                                    probe_no_spaces = device_name.replace(" ", "")
                                    if probe_no_spaces in stream_name:
                                        probe_device_name = device_name
                                        electrode_group_location = target_locations.get(device_name, "unknown")
                                        probe_device = device
                                        logging.info(
                                            f"Found device from rig: {device_name} at location {electrode_group_location}"
                                        )
                                        break

                            # if probe_device_name not found in metadata, use probes_info from recording
                            if probe_device_name is None:
                                # if devices_from_rig not found in metadata, use probes_info from recording
                                electrode_group_location = "unknown"
                                probes_info = recording.get_annotation("probes_info", None)
                                if probes_info is not None and len(probes_info) == 1:
                                    probe_info = probes_info[0]
                                    probe_device_name = probe_info.get("name", None)
                                    probe_device_manufacturer = probe_info.get("manufacturer", None)
                                    probe_model_name = probe_info.get("model_name", None)
                                    probe_serial_number = probe_info.get("serial_number", None)
                                    probe_device_description = ""
                                    if probe_device_name is None:
                                        if probe_model_name is not None:
                                            probe_device_name = probe_model_name
                                        else:
                                            probe_device_name = "Probe"
                                    if probe_model_name is not None:
                                        probe_device_description += f"Model: {probe_device_description}"
                                    if probe_serial_number is not None:
                                        if len(probe_device_description) > 0:
                                            probe_device_description += " - "
                                        probe_device_description += f"Serial number: {probe_serial_number}"
                                    probe_device = Device(
                                        name=probe_device_name,
                                        description=probe_device_description,
                                        manufacturer=probe_device_manufacturer,
                                    )
                                else:
                                    logging.info("\tCould not load device information: using default Device")
                                    probe_device_name = "Device"
                                    probe_device = Device(name=probe_device_name, description="Default device")

                                if probe_device_name not in nwbfile.devices:
                                    nwbfile.add_device(probe_device)
                                    logging.info(f"\tAdded probe device: {probe_device.name} from probeinterface")

                            electrode_metadata = dict(
                                Ecephys=dict(
                                    Device=[dict(name=probe_device_name)],
                                )
                            )

                            if (postprocessed_folder / f"{recording_name}.zarr").is_dir():
                                # zarr format
                                analyzer_folder = postprocessed_folder / f"{recording_name}.zarr"
                            else:
                                # binary format
                                analyzer_folder = postprocessed_folder / recording_name

                            analyzer = si.load_sorting_analyzer(analyzer_folder, load_extensions=False)

                            # Load curated sorting and set properties
                            sorting_curated = si.load(curated_folder / recording_name)

                            # Add unit properties (UUID and probe info, ks_unit_id)
                            unit_uuids = [str(uuid4()) for u in sorting_curated.unit_ids]
                            sorting_curated.set_property(
                                "device_name", [probe_device_name] * sorting_curated.get_num_units()
                            )
                            sorting_curated.set_property("shank", [group_str] * sorting_curated.get_num_units())
                            sorting_curated.set_property("unit_name", unit_uuids)
                            sorting_curated.set_property("ks_unit_id", sorting_curated.unit_ids)

                            # Add 'amplitude' property
                            amplitudes = np.round(list(si.get_template_extremum_amplitude(analyzer, mode="peak_to_peak").values()), 2)
                            sorting_curated.set_property("amplitude", amplitudes)
                            # Add depth property
                            unit_locations = np.round(analyzer.get_extension("unit_locations").get_data(), 2)
                            sorting_curated.set_property("estimated_x", unit_locations[:, 0])
                            sorting_curated.set_property("estimated_y", unit_locations[:, 1])
                            if unit_locations.shape[1] == 3:
                                sorting_curated.set_property("estimated_z", unit_locations[:, 2])
                            sorting_curated.set_property("depth", unit_locations[:, 1])
                            logging.info(f"\tAdding {len(sorting_curated.unit_ids)} units from stream {stream_name}")

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

                            recording_list = recordings_associated_to_recording_name[recording_name]
                            if aggregate_groups[(block_str, recording_str)]:
                                # Add channel properties (group_name property to associate electrodes with group)
                                recording_all = si.aggregate_channels(recording_list)
                                if aggregate_groups[(block_str, recording_str)]:
                                    logging.info(f"Aggregating {len(recording_list)} for {recording_name}")
                                    recording = recording_all

                            if len(recording_list) == 1:
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
                                channel_groups = recording.get_channel_groups()
                                recording.set_channel_groups([f"{probe_device_name}_group{g}" for g in channel_groups])
                                channel_groups = np.unique(recording.get_channel_groups())
                                electrode_groups_metadata = [
                                    dict(
                                        name=f"{probe_device_name}_group{g}",
                                        description=f"Recorded electrodes from probe {g}",
                                        location=electrode_group_location,
                                        device=probe_device_name,
                                    )
                                    for g in channel_groups
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
