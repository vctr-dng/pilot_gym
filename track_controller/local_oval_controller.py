from track import Track

from .local_list_controller import LocalListController


class LocalOvalController(LocalListController):
    def __init__(self, track_list: list[Track], generate_if_missing: bool = True):
        super().__init__(track_list)
        self.generate_if_missing = generate_if_missing

    def get_track(self, index):
        try:
            return super().get_track(index)
        except IndexError as e:
            if self.generate_if_missing:
                self.generate_track(index)
            else:
                raise e
