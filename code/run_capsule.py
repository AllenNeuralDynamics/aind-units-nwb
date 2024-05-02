""" Writes Units to an NWB file """
import shutil
import json
from pathlib import Path
import numpy as np

from uuid import uuid4

import probeinterface as pi
import spikeinterface as si
# needed to lead extensions
import spikeinterface.extractors as se
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm

from pynwb import NWBHDF5IO
from pynwb.file import Device
from hdmf_zarr import NWBZarrIO

from utils import get_devices_from_metadata, add_waveforms_with_uneven_channels


data_folder = Path("../data")
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
        p for p in data_folder.glob("**/*") 
        if (p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr"))
        and ("ecephys_" in p.name or "behavior_" in p.name)
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
        if p.is_dir() and ("ecephys" in p.name or "behavior" in p.name) and ("sorted" not in p.name and "nwb" not in p.name)
    ]
    assert len(ecephys_folders) == 1, "Attach one ecephys folder at a time"
    ecephys_folder = ecephys_folders[0]
    ecephys_compressed_folder = ecephys_folder / "ecephys_compressed"
    ecephys_clipped_folder = ecephys_folder / "ecephys_clipped"
    if ecephys_compressed_folder.is_dir():
        compressed = True
        open_ephys_folder = ecephys_clipped_folder
    else:
        compressed = False
        open_ephys_folder = ecephys_folder / "ecephys"

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
        error_txt.write_text(
            "Postprocessed folder not found. No NWB files were created."
        )
    else:
        assert curated_folder.is_dir(), f"Curated folder {curated_folder} does not exist"
        assert spikesorted_folder.is_dir(), f"Spikesorted folder {spikesorted_folder} does not exist"

        # we create a result NWB file for each experiment/recording
        recording_names = [p.name for p in curated_folder.iterdir() if p.is_dir()]

        # find experiment and recording ids
        experiment_ids = []
        recording_ids = []
        stream_names = []
        for recording_name in recording_names:
            experiment_str = recording_name.split("_")[0]
            recording_str = recording_name.split("_")[-1]
            stream_name = "_".join(recording_name.split("_")[1:-1])
            experiment_id = int(experiment_str[len("experiment") :])
            recording_id = int(recording_str[len("recording") :])
            if experiment_id not in experiment_ids:
                experiment_ids.append(experiment_id)
            if recording_id not in recording_ids:
                recording_ids.append(recording_id)
            if stream_name not in stream_names:
                stream_names.append(stream_name)

        nwb_output_files = []
        for block_index, experiment_id in enumerate(experiment_ids):
            for segment_index, recording_id in enumerate(recording_ids):
                # add recording/experiment id if needed
                nwb_original_file_name = nwbfile_input_path.stem
                if "experiment" in nwb_original_file_name and "recording" in nwb_original_file_name:
                    nwb_file_name = f"{nwb_original_file_name}.nwb"
                else:
                    nwb_file_name = (
                        f"{nwb_original_file_name}_experiment{experiment_id}_recording{recording_id}.nwb"
                    )
                nwbfile_output_path = output_folder / nwb_file_name

                # Find probe devices
                devices, target_locations = get_devices_from_metadata(ecephys_folder, segment_index=segment_index)

                with io_class(str(nwbfile_input_path), "r") as read_io:
                    nwbfile = read_io.read()

                    added_stream_names = []
                    for stream_name in stream_names:
                        recording_name = f"experiment{experiment_id}_{stream_name}_recording{recording_id}"
                        if not (curated_folder / recording_name).is_dir():
                            print(
                                f"Curated units for stream {stream_name} for experiment "
                                f"{experiment_id} and recording {recording_id} not found."
                            )
                            continue

                        added_stream_names.append(stream_name)

                        # Read Zarr recording
                        if not compressed:
                            recording = se.read_openephys(
                                ecephys_folder,
                                stream_name=stream_name,
                                block_index=block_index,
                                load_sync_timestamps=True
                            )
                        else:
                            recording = si.read_zarr(ecephys_compressed_folder / f"experiment{experiment_id}_{stream_name}.zarr")

                        # Load synchronized timestamps and attach to recording
                        record_node, oe_stream_name = stream_name.split("#")
                        recording_folder = open_ephys_folder / record_node
                        recording = si.split_recording(recording)[segment_index]
                        # set times as np.array to speed up spike train retrieval later
                        recording.set_times(np.array(recording.get_times()))

                        # Add device and electrode group
                        if devices:
                            for device_name, device in devices.items():
                                probe_no_spaces = device_name.replace(" ", "")
                                if probe_no_spaces in oe_stream_name:
                                    if device_name not in nwbfile.devices:
                                        nwbfile.add_device(devices[device_name])
                                    probe_device_name = probe_no_spaces
                                    if device_name in target_locations:
                                        electrode_group_location = target_locations[device_name]
                                    else:
                                        electrode_group_location = "unknown"
                                    print(f"Found device from rig: {probe_device_name}")
                                    break
                        else:
                            # if devices not found in metadata, instantiate using probeinterface
                            electrode_group_location = "unknown"
                            if experiment_id == 1:
                                settings_file = recording_folder / "settings.xml"
                            else:
                                settings_file = recording_folder / f"settings_{experiment_id}.xml"
                            probe = pi.read_openephys(settings_file, stream_name=oe_stream_name)
                            probe_device_name = probe.name
                            probe_device_description = f"Model: {probe.model_name} - Serial number: {probe.serial_number}"
                            probe_device_manufacturer = f"{probe.manufacturer}"
                            probe_device = Device(
                                name=probe_device_name,
                                description=probe_device_description,
                                manufacturer=probe_device_manufacturer,
                            )
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

                        we = si.load_waveforms(
                            postprocessed_folder / recording_name, with_recording=False
                        )

                        # Load curated sorting and set properties
                        sorting_curated = si.load_extractor(curated_folder / recording_name)

                        # Add unit properties (UUID and probe info, ks_unit_id)
                        unit_uuids = [str(uuid4()) for u in sorting_curated.unit_ids]
                        sorting_curated.set_property("device_name", [probe_device_name] * sorting_curated.get_num_units())
                        sorting_curated.set_property("unit_name", unit_uuids)
                        sorting_curated.set_property("ks_unit_id", sorting_curated.unit_ids)

                        # Add 'amplitude' property
                        amplitudes = np.round(list(si.get_template_extremum_amplitude(we).values()), 2)
                        sorting_curated.set_property("amplitude", amplitudes)
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

                    with io_class(str(nwbfile_output_path), "w") as export_io:
                        export_io.export(src_io=read_io, nwbfile=nwbfile)
                    print(f"Done writing {nwbfile_output_path}")
                    nwb_output_files.append(nwbfile_output_path)
