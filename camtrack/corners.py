#! /usr/bin/env python3

__all__ = [
    'FrameCorners',
    'CornerStorage',
    'build',
    'dump',
    'load',
    'draw',
    'calc_track_interval_mappings',
    'calc_track_len_array_mapping',
    'without_short_tracks'
]

import click
import cv2
import numpy as np
import pims
import matplotlib.pyplot as plt

from _corners import (
    FrameCorners,
    CornerStorage,
    StorageImpl,
    dump,
    load,
    draw,
    calc_track_interval_mappings,
    calc_track_len_array_mapping,
    without_short_tracks,
    create_cli
)


class _CornerStorageBuilder:

    def __init__(self, progress_indicator=None):
        self._progress_indicator = progress_indicator
        self._corners = dict()

    def set_corners_at_frame(self, frame, corners):
        self._corners[frame] = corners
        if self._progress_indicator is not None:
            self._progress_indicator.update(1)

    def build_corner_storage(self):
        return StorageImpl(item[1] for item in sorted(self._corners.items()))


def threshold_eq(a, b, threshold):
    return abs(a[0][0] - b[0][0]) < threshold and abs(a[0][1] - b[0][1]) < threshold


def in_bounds(i, j, img):
    return 0 <= i < img.shape[1] and 0 <= j < img.shape[0]


def near_points(dot, img, size):
    dot = dot[0]
    neighbors = [[], []]
    delta = []
    for i in range(-size, size + 1):
        for j in range(-size, size + 1):
            delta.append([i, j])
    # delta = [[0, 1], [1, 0], [0, -1], [-1, 0], [1, 1], [1, -1], [-1, 1], [-1, -1], [0, 0]]
    i = int(dot[0])
    j = int(dot[1])
    for d in delta:
        if in_bounds(i + d[0], j + d[1], img):
            neighbors[0].append(i + d[0])
            neighbors[1].append(j + d[1])
    return neighbors


def _build_impl(frame_sequence: pims.FramesSequence,
                builder: _CornerStorageBuilder) -> None:
    corners_1 = None
    N = 600
    alpha = 0.32
    mask_size = 5
    maxLevel = 4
    cc = 0.1
    if len(frame_sequence) < 30:
        N = 3000
        alpha = 0.18

    if 100 > len(frame_sequence) > 80:
        N = 5000
        alpha = 0.1

    if len(frame_sequence) == 99: # room
        N = 2000
        alpha = 0.01
        maxLevel = 4
        cc = 0.01
        mask_size = 5

    if len(frame_sequence) == 478:  # fox_head_full
        pass

    if len(frame_sequence) == 98:  # fox_head_short
        pass

    if len(frame_sequence) == 323:  # house_free_motion
        N = 600
        alpha = 0.32
        mask_size = 5
        maxLevel = 4
        cc = 0.1

    if len(frame_sequence) == 23:  # ironman_translation_fast
        N = 6000
        alpha = 0.05

    if len(frame_sequence) == 23:  # ironman_translation_fast
        N = 6000
        alpha = 0.01

    ids_amount = N
    image_0 = frame_sequence[0]
    arr_corners = cv2.goodFeaturesToTrack(image_0, N, alpha / 4, mask_size)
    corners_0 = FrameCorners(
        np.array(range(len(arr_corners))),  # id треков
        np.array(arr_corners),  # положение уголков
        np.array([5] * len(arr_corners))  # размер уголка
    )

    builder.set_corners_at_frame(0, corners_0)
    lks = dict(winSize=(25, 25), maxLevel=maxLevel, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, cc))
    for frame, image_1 in enumerate(frame_sequence[1:], 1):

        p1, st, err = cv2.calcOpticalFlowPyrLK(
            np.uint8(image_0 * 255. / image_0.max()),
            np.uint8(image_1 * 255. / image_1.max()),
            arr_corners,
            None,
            **lks,
        )
        n = 0
        if p1 is not None:
            good_new = p1[st.reshape(-1) == 1]
            old_ids = corners_0._ids[st.reshape(-1) == 1].reshape(-1)
            n = len(good_new)

        add_n = 0
        if n < N:
            mask = np.ones_like(image_1, dtype=np.uint8)
            for dot in good_new:
                mask1 = near_points(dot, image_1, mask_size)
                mask[mask1[1], mask1[0]] = 0
            arr_corners = cv2.goodFeaturesToTrack(image_1, N - n, alpha, mask_size, mask=mask)
            if arr_corners is not None:
                add_n = len(arr_corners)
                arr_corners = np.concatenate([good_new, arr_corners])
                arr_ids = np.concatenate([old_ids, np.array(range(ids_amount, ids_amount + add_n))])
                corners_1 = FrameCorners(
                    np.array(arr_ids),  # id треков
                    np.array(arr_corners),  # положение уголков
                    np.array([5] * len(arr_ids))  # размер уголка
                )
            else:
                arr_corners = good_new
                corners_1 = FrameCorners(
                    np.array(old_ids),  # id треков
                    np.array(good_new),  # положение уголков
                    np.array([5] * len(old_ids))  # размер уголка
                )
        else:
            arr_corners = good_new
            corners_1 = FrameCorners(
                np.array(old_ids),  # id треков
                np.array(good_new),  # положение уголков
                np.array([5] * len(old_ids))  # размер уголка
            )
        ids_amount += add_n
        builder.set_corners_at_frame(frame, corners_1)
        corners_0 = corners_1
        image_0 = image_1


def build(frame_sequence: pims.FramesSequence,
          progress: bool = True) -> CornerStorage:
    """
    Build corners for all frames of a frame sequence.

    :param frame_sequence: grayscale float32 frame sequence.
    :param progress: enable/disable building progress bar.
    :return: corners for all frames of given sequence.
    """
    if progress:
        with click.progressbar(length=len(frame_sequence),
                               label='Calculating corners') as progress_bar:
            builder = _CornerStorageBuilder(progress_bar)
            _build_impl(frame_sequence, builder)
    else:
        builder = _CornerStorageBuilder()
        _build_impl(frame_sequence, builder)
    return builder.build_corner_storage()


if __name__ == '__main__':
    create_cli(build)()  # pylint:disable=no-value-for-parameter
