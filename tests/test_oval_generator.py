# %%
from track_generator import OvalGenerator


def test_oval_generator():
    oval_settings = {
        "straight_length": 2,
        "track_width": 3,
        "turning_radius": 8,
        "sampling_rate": 2,
    }

    oval_generator = OvalGenerator(**oval_settings)
    track = oval_generator.generate_track()
    track.plot()
    track.save(
        f"data/oval_track_{oval_settings['straight_length']}_{oval_settings['track_width']}_{oval_settings['turning_radius']}_{oval_settings['sampling_rate']}.npz"
    )


if __name__ == "__main__":
    test_oval_generator()
# %%
