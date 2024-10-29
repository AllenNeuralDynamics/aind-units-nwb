""" Writes Units to an NWB file """
import shutil
import json
from pathlib import Path
import numpy as np

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
    print("\n\nNWB EXPORT UNITS")
    t_export_start = time.perf_counter()

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
    print(f"NWB backend: {NWB_BACKEND}")

    # if more than 1 input NWB files, we copy them all to the results
    # since some processing might have failed
    if len(nwb_files) > 1:
        for nwb_file_path in nwb_files:
            if nwb_file_path.is_dir():
                shutil.copytree(nwb_file_path, results_folder / nwb_file_path.name)
            else:
                shutil.copyfile(nwb_file_path, results_folder / nwb_file_path.name)

    # find raw data
    ecephys_folders = [
        p
        for p in data_folder.iterdir()
        if p.is_dir()
        and ("ecephys" in p.name or "behavior" in p.name)
        and "sorted" not in p.name and "nwb" not in p.name
    ]
    assert len(ecephys_folders) == 1, "Attach one ecephys folder at a time"
    ecephys_folder = ecephys_folders[0]

    # find raw data
    job_json_files = [p for p in data_folder.iterdir() if p.suffix == ".json" and "job" in p.name]
    job_dicts = []
    recording_names_in_json = []
    for job_json_file in job_json_files:
        with open(job_json_file) as f:
            job_dict = json.load(f)
            recording_names_in_json.append(job_dict["recording_name"])
        job_dicts.append(job_dict)
    print(f"Found {len(job_dicts)} JSON job files. Recording names:\n{recording_names_in_json}")

    # find sorted data
    sorted_folders = [
        p for p in data_folder.iterdir() if p.is_dir() and "sorted" in p.name and "spikesorted" not in p.name
    ]

    if len(sorted_folders) == 0:
        # pipeline mode
        if (data_folder / "collect_pipeline_output_test").is_dir():
            print("\n*******************\n**** TEST MODE ****\n*******************\n")
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
        print("Postprocessed folder not found. Skipping NWB export")
        # create dummy nwb folder to avoid pipeline failure
        error_txt = results_folder / "error.txt"
        error_txt.write_text("Postprocessed folder not found. No NWB files were created.")
    else:
        assert curated_folder.is_dir(), f"Curated folder {curated_folder} does not exist"
        assert spikesorted_folder.is_dir(), f"Spikesorted folder {spikesorted_folder} does not exist"

        # we create a result NWB file for each experiment/recording
        recording_names = [p.name for p in curated_folder.iterdir() if p.is_dir()]

        # find blocks and recordings
        block_ids = []
        recording_ids = []
        group_ids = []
        stream_names = []
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
                group_str = None

            if block_str not in block_ids:
                block_ids.append(block_str)
            if recording_str not in recording_ids:
                recording_ids.append(recording_str)
            if stream_name not in stream_names:
                stream_names.append(stream_name)
            if group_str is not None and group_str not in group_ids:
                group_ids.append(group_str)
        if len(group_ids) == 0:
            group_ids = [""]

        block_ids = sorted(block_ids)
        recording_ids = sorted(recording_ids)
        stream_names = sorted(stream_names)

        print(f"Number of NWB files to write: {len(block_ids) * len(recording_ids)}")
        print(f"Number of streams to write for each file: {len(stream_names)}")

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
                        p for p in nwb_files if block_str in p.stem and recording_str in p.stem
                    ]
                    assert len(nwb_input_path_for_current) == 1, (
                        f"Could not find input NWB file for {block_str}-{recording_str}. Available NWB files are: {nwb_files}"
                    )
                    nwbfile_input_path = nwb_input_path_for_current[0]
                    print(f"Found input NWB file for {block_str}-{recording_str}: {nwbfile_input_path.name}")
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
                    ecephys_folder, segment_index=segment_index
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
                                print(f"Curated units for {recording_name} not found.")
                                continue

                            # load JSON and recordings
                            recording_job_dict = None
                            for job_dict in job_dicts:
                                if job_dict["recording_name"] == recording_name:
                                    recording_job_dict = job_dict
                                    break
                            if recording_job_dict is None:
                                print(f"Could not find JSON file associated to {recording_name}")
                                continue

                            added_stream_names.append(stream_str)

                            recording = si.load_extractor(job_dict["recording_dict"], base_folder=data_folder)
                            skip_times = job_dict.get("skip_times", False)
                            if skip_times:
                                recording.reset_times()

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
                                        print(
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
                                    print("\tCould not load device information: using default Device")
                                    probe_device_name = "Device"
                                    probe_device = Device(name=probe_device_name, description="Default device")

                                if probe_device_name not in nwbfile.devices:
                                    nwbfile.add_device(probe_device)
                                    print(f"\tAdded probe device: {probe_device.name} from probeinterface")

                            electrode_metadata = dict(
                                Ecephys=dict(
                                    Device=[dict(name=probe_device_name)],
                                    ElectrodeGroup=[
                                        dict(
                                            name=probe_device_name,
                                            description=f"Recorded electrodes from probe {probe_device_name}",
                                            location=electrode_group_location,
                                            device=probe_device_name,
                                        )
                                    ],
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
                            sorting_curated = si.load_extractor(curated_folder / recording_name)

                            # Add unit properties (UUID and probe info, ks_unit_id)
                            unit_uuids = [str(uuid4()) for u in sorting_curated.unit_ids]
                            sorting_curated.set_property(
                                "device_name", [probe_device_name] * sorting_curated.get_num_units()
                            )
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
                            print(f"\tAdding {len(sorting_curated.unit_ids)} units from stream {stream_name}")

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
                            # set channel groups to match previously added ephys electrodes
                            recording.set_channel_groups([probe_device_name] * recording.get_num_channels())
                            add_waveforms_with_uneven_channels(
                                sorting_analyzer=analyzer,
                                recording=recording,
                                nwbfile=nwbfile,
                                metadata=electrode_metadata,
                                skip_properties=skip_unit_properties,
                                units_description=units_description,
                            )
                    print(f"Added {len(added_stream_names)} streams")

                    if NWB_BACKEND == "zarr":
                        write_args = {'link_data': False}
                    else:
                        write_args = {}

                    t_write_start = time.perf_counter()
                    append_io.write(nwbfile)
                    t_write_end = time.perf_counter()
                    elapsed_time_write = np.round(t_write_end - t_write_start, 2)
                    print(f"Writing time: {elapsed_time_write}s")
                    # with io_class(str(nwbfile_output_path), "w") as export_io:
                    #    export_io.export(src_io=read_io, nwbfile=nwbfile, write_args=write_args)
                    print(f"Done writing {nwbfile_output_path}")
                    nwb_output_files.append(nwbfile_output_path)

    t_export_end = time.perf_counter()
    elapsed_time_export = np.round(t_export_end - t_export_start, 2)
    print(f"NWB EXPORT UNITS time: {elapsed_time_export}s")
