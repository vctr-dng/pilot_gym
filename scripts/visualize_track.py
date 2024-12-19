# %%
from track.track import Track

track_path = "../data/tracks/oval_track_40_8_15_2.npz"

track = Track.load(track_path)
track.plot()
# %%
