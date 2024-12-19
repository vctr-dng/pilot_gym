import numpy as np

from track import Track

from .track_generator import TrackGenerator


class OvalGenerator(TrackGenerator):
    def __init__(
        self,
        straight_length: float,
        track_width: float,
        turning_radius: float,
        sampling_rate: float,
        origin: np.array = np.array([0, 0]),
    ):
        super().__init__(sampling_rate)
        self.straight_length = straight_length
        self.track_width = track_width
        self.turning_radius = turning_radius
        self.origin = origin

    def generate_oval_edge(self, width_offset: float = 0):
        new_origin = self.origin + np.array([width_offset, 0])

        first_straight = self.generate_straight(
            start_point=new_origin,
            end_point=new_origin + np.array([0.0, self.straight_length]),
        )

        last_straight = self.generate_straight(
            start_point=new_origin
            + np.array(
                [2 * (self.turning_radius - width_offset), self.straight_length]
            ),
            end_point=new_origin
            + np.array([2 * (self.turning_radius - width_offset), 0]),
        )

        first_hairpin = self.generate_corner(
            center=self.origin + np.array([self.turning_radius, self.straight_length]),
            radius=self.turning_radius - width_offset,
            start_angle=np.pi,
            end_angle=0,
        )

        last_hairpin = self.generate_corner(
            center=self.origin + np.array([self.turning_radius, 0]),
            radius=self.turning_radius - width_offset,
            start_angle=0,
            end_angle=-np.pi,
        )

        oval_edge = np.vstack(
            (
                first_straight[:-1],
                first_hairpin[:-1],
                last_straight[:-1],
                last_hairpin[:-1],
            )
        )

        return oval_edge

    def generate_track(self) -> Track:
        center_line = self.generate_oval_edge()
        left_boundaries = self.generate_oval_edge(width_offset=-self.track_width / 2)
        right_boundaries = self.generate_oval_edge(width_offset=self.track_width / 2)

        return Track(
            right_boundaries=right_boundaries,
            left_boundaries=left_boundaries,
            center_line=center_line,
            reference_path=center_line,
            closed_loop=True,
        )
