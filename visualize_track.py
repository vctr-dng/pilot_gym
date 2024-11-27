# %%

from track.track import Track

track_path = "data/tracks/oval_track_100_2_8_2.npz"

track = Track.load(track_path)
track.plot()
# %%
