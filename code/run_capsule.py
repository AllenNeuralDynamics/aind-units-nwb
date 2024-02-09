""" Writes Units to an NWB file """
from pathlib import Path
import numpy as np

from uuid import uuid4

import probeinterface as pi
import spikeinterface as si
# needed to lead extensions
import spikeinterface.postprocessing as spost
import spikeinterface.qualitymetrics as sqm

from neuroconv.tools.spikeinterface.spikeinterface import add_waveforms

from pynwb import NWBHDF5IO
from pynwb.file import Device
from hdmf_zarr import NWBZarrIO

from utils import get_devices_from_metadata


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
    nwb_files = [p for p in data_folder.glob("**/*") if p.name.endswith(".nwb") or p.name.endswith(".nwb.zarr")]
    assert len(nwb_files) == 1, "Attach one base NWB file data at a time"
    nwbfile_input_path = nwb_files[0]

    if nwbfile_input_path.is_dir():
        assert (nwbfile_input_path / ".zattrs").is_file(), f"{nwbfile_input_path.name} is not a valid Zarr folder"
        NWB_BACKEND = "zarr"
        NWB_SUFFIX = ".nwb.zarr"
        io_class = NWBZarrIO
    else:
        NWB_BACKEND = "hdf5"
        NWB_SUFFIX = ".nwb"
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
    assert postprocessed_folder.is_dir(), f"Postprocessed folder {postprocessed_folder} does not exist"
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
            nwbfile_output_path = output_folder / nwbfile_input_path.name

            # Find probe devices
            devices, target_locations = get_devices_from_metadata(ecephys_folder, segment_index=segment_index)

            with io_class(str(nwbfile_input_path), "r") as read_io:
                nwbfile = read_io.read()

                added_stream_names = []
                for stream_name in stream_names:
                    recording_name = f"experiment{experiment_id}_{stream_name}_recording{recording_id}"
                    if not (curated_folder / recording_name).is_dir():
                        print(
                            f"Stream {stream_name} for experiment {experiment_id} and recording {recording_id} does not exist"
                        )
                        continue

                    added_stream_names.append(stream_name)

                    # Read Zarr recording
                    zarr_folder = ecephys_compressed_folder / f"experiment{experiment_id}_{stream_name}.zarr"
                    recording = si.load_extractor(zarr_folder)

                    # Load synchronized timestamps and attach to recording
                    record_node, oe_stream_name = stream_name.split("#")
                    recording_folder = ecephys_clipped_folder / record_node
                    stream_folder = (
                        recording_folder
                        / f"experiment{experiment_id}"
                        / f"recording{recording_id}"
                        / "continuous"
                        / oe_stream_name
                    )
                    if (stream_folder / "sample_numbers.npy").is_file():
                        # version>=v0.6
                        sync_times = np.load(stream_folder / "timestamps.npy")
                    else:
                        # version<v0.6
                        sync_times = np.load(stream_folder / "synchronized_timestamps.npy")
                    recording = si.split_recording(recording)[segment_index]

                    if len(sync_times) == recording.get_num_samples():
                        original_times = recording.get_times()
                        recording.set_times(sync_times, with_warning=False)
                    else:
                        print(
                            f"recording{segment_index+1}: mismatch between num samples ({recording.get_num_samples()}) and timestamps ({len(sync_times)})"
                        )

                    sorting_curated = si.load_extractor(curated_folder / recording_name)
                    # register recording for precise timestamps
                    sorting_curated.register_recording(recording)
                    we = si.load_waveforms(
                        postprocessed_folder / recording_name, sorting=sorting_curated, with_recording=False
                    )
                    # Add 'amplitude' property
                    amplitudes = np.round(list(si.get_template_extremum_amplitude(we).values()), 2)
                    sorting_curated.set_property("amplitude", amplitudes)
                    print(f"\tAdding {len(sorting_curated.unit_ids)} units from stream {stream_name}")

                    # Add property for channel quality
                    good_channel_mask = np.isin(recording.channel_ids, we.channel_ids)
                    channel_quality = np.array(["good"] * len(recording.channel_ids))
                    channel_quality[~good_channel_mask] = "bad"
                    recording.set_property("quality", channel_quality)

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

                    # Add unit properties (UUID and probe info, ks_unit_id)
                    unit_uuids = [str(uuid4()) for u in sorting_curated.unit_ids]
                    sorting_curated.set_property("device_name", [probe_device_name] * sorting_curated.get_num_units())
                    sorting_curated.set_property("unit_name", unit_uuids)
                    sorting_curated.set_property("ks_unit_id", sorting_curated.unit_ids)

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

                    # Add channel properties (group_name property to associate electrodes with group)
                    recording.set_channel_groups([probe_device_name] * recording.get_num_channels())

                    we.sorting = sorting_curated

                    # print("NWB metadata:\n", electrode_metadata)
                    print("\tAdding sorting and waveforms data")
                    add_waveforms(
                        we,
                        nwbfile=nwbfile,
                        metadata=electrode_metadata,
                        skip_properties=skip_unit_properties,
                        units_description="Units from Kilosort2.5",
                        recording=recording,
                    )

                print(f"Added {len(added_stream_names)} streams")

                with io_class(str(nwbfile_output_path), "w") as export_io:
                    export_io.export(src_io=read_io, nwbfile=nwbfile)
                print(f"Done writing {nwbfile_output_path}")
                nwb_output_files.append(nwbfile_output_path)
