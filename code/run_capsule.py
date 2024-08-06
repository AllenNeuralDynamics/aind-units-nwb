""" Writes Units to an NWB file """
import shutil
import json
from pathlib import Path
import numpy as np

from uuid import uuid4

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
    # find base NWB file
    nwb_files = [
        p
        for p in data_folder.glob("**/*")
        if (p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")) and ("ecephys_" in p.name or "behavior_" in p.name) and "/nwb/" not in str(p)
    ]
    assert len(nwb_files) == 1, "Attach one base NWB file data at a time"
    nwbfile_input_path = nwb_files[0]

    if nwbfile_input_path.is_dir():
        assert (nwbfile_input_path / ".zattrs").is_file(), f"{nwbfile_input_path.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        io_class = NWBZarrIO
    else:
        NWB_BACKEND = "hdf5"
        io_class = NWBHDF5IO
    print(f"NWB backend: {NWB_BACKEND}")

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
    for job_json_file in job_json_files:
        with open(job_json_file) as f:
            job_dict = json.load(f)
        job_dicts.append(job_dict)
    print(f"Found {len(job_dicts)} JSON job files")

    # find sorted data
    sorted_folders = [
        p for p in data_folder.iterdir() if p.is_dir() and "sorted" in p.name and "spikesorted" not in p.name
    ]

    if len(sorted_folders) == 0:
        # pipeline mode
        sorted_folder = data_folder
        output_folder = results_folder / "nwb"
        output_folder.mkdir(exist_ok=True)
    elif len(sorted_folders) == 1:
        # capsule mode
        sorted_folder = sorted_folders[0]
        output_folder = results_folder

    postprocessed_folder = sorted_folder / "postprocessed"
    curated_folder = sorted_folder / "curated"
    spikesorted_folder = sorted_folder / "spikesorted"
    if not postprocessed_folder.is_dir():
        print("Postprocessed folder not found. Skipping NWB export")
        # create dummy nwb folder to avoid pipeline failure
        error_txt = output_folder / "error.txt"
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

        nwb_output_files = []
        for block_index, block_str in enumerate(block_ids):
            for segment_index, recording_str in enumerate(recording_ids):
                # add recording/experiment id if needed
                nwb_original_file_name = nwbfile_input_path.stem
                if block_str in nwb_original_file_name and recording_str in nwb_original_file_name:
                    nwb_file_name = f"{nwb_original_file_name}.nwb"
                else:
                    nwb_file_name = f"{nwb_original_file_name}_{block_str}_{recording_str}.nwb"
                nwbfile_output_path = output_folder / nwb_file_name

                # copy to results to avoid read-only issues
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
                            # set times as np.array to speed up spike train retrieval later
                            recording.set_times(np.array(recording.get_times()))

                            # Add device and electrode group
                            probe_device_name = None
                            if devices_from_rig:
                                for device_name, device in devices_from_rig.items():
                                    # add the device, since it could be a laser
                                    if device_name not in nwbfile.devices:
                                        nwbfile.add_device(devices_from_rig[device_name])
                                    # find probe device name
                                    probe_no_spaces = device_name.replace(" ", "")
                                    if probe_no_spaces in stream_name:
                                        probe_device_name = probe_no_spaces
                                        electrode_group_location = target_locations.get(device_name, "unknown")
                                        print(
                                            f"Found device from rig: {probe_device_name} at location {electrode_group_location}"
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

                            we = si.load_waveforms(postprocessed_folder / recording_name, with_recording=False)

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
                            amplitudes = np.round(list(si.get_template_extremum_amplitude(we).values()), 2)
                            sorting_curated.set_property("amplitude", amplitudes)

                            # Add depth property
                            unit_locations = np.round(we.load_extension("unit_locations").get_data(), 2)
                            sorting_curated.set_property("estimated_x", unit_locations[:, 0])
                            sorting_curated.set_property("estimated_y", unit_locations[:, 1])
                            if unit_locations.shape[1] == 3:
                                sorting_curated.set_property("estimated_z", unit_locations[:, 2])
                            sorting_curated.set_property("depth", unit_locations[:, 1])
                            print(f"\tAdding {len(sorting_curated.unit_ids)} units from stream {stream_name}")

                            # Register recording for precise timestamps
                            sorting_curated.register_recording(recording)
                            we.sorting = sorting_curated

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
                                waveform_extractor=we,
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

                    append_io.write(nwbfile)
                    # with io_class(str(nwbfile_output_path), "w") as export_io:
                    #    export_io.export(src_io=read_io, nwbfile=nwbfile, write_args=write_args)
                    print(f"Done writing {nwbfile_output_path}")
                    nwb_output_files.append(nwbfile_output_path)
