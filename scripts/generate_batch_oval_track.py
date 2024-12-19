import numpy as np
import yaml

from track_generator import OvalGenerator


def main():
    straight_length_sweep = np.arange(40, 41)
    track_width_sweep = np.arange(8, 9)
    turning_radius_sweep = np.arange(15, 16)
    sample_rate = 2

    saved_paths = []
    generated_tracks = []
    for straight_length in straight_length_sweep:
        for track_width in track_width_sweep:
            for turning_radius in turning_radius_sweep:
                oval_settings = {
                    "straight_length": straight_length,
                    "track_width": track_width,
                    "turning_radius": turning_radius,
                    "sampling_rate": sample_rate,
                }
                oval_generator = OvalGenerator(**oval_settings)
                track = oval_generator.generate_track()
                generated_tracks.append(track)
                save_path = (
                    f"data/tracks/oval_track"
                    f"_{oval_settings['straight_length']}"
                    f"_{oval_settings['track_width']}"
                    f"_{oval_settings['turning_radius']}"
                    f"_{oval_settings['sampling_rate']}.npz"
                )
                track.save(save_path)
                saved_paths.append(save_path)

    # Save in a yaml file the saved_paths
    with open("data/track_list.yaml", "w") as file:
        yaml.dump(saved_paths, file)
    
    for track in generated_tracks:
        track.plot()


if __name__ == "__main__":
    main()
