#! /usr/bin/env python3

__all__ = [
    'track_and_calc_colors'
]

from typing import List, Optional, Tuple

import cv2
import numpy as np

from corners import CornerStorage
from data3d import CameraParameters, PointCloud, Pose
import frameseq
from _camtrack import (
    PointCloudBuilder,
    create_cli,
    calc_point_cloud_colors,
    pose_to_view_mat3x4,
    to_opencv_camera_mat3x3,
    view_mat3x4_to_pose,
    build_correspondences,
    triangulate_correspondences,
    TriangulationParameters,
    rodrigues_and_translation_to_view_mat3x4, view_mat3x4_to_rodrigues_and_translation
)


def triangulate_nviews(mats, points):
    n = len(mats)
    solve_mat = np.zeros([3 * n, 4 + n])
    for i, (x, p) in enumerate(zip(points, mats)):
        solve_mat[3 * i:3 * i + 3, :4] = p
        solve_mat[3 * i:3 * i + 3, 4 + i] = -x
    A = np.linalg.svd(solve_mat)[-1][-1, :4]
    return A / A[3]


def track_and_calc_colors(camera_parameters: CameraParameters,
                          corner_storage: CornerStorage,
                          frame_sequence_path: str,
                          known_view_1: Optional[Tuple[int, Pose]] = None,
                          known_view_2: Optional[Tuple[int, Pose]] = None) \
        -> Tuple[List[Pose], PointCloud]:
    if known_view_1 is None or known_view_2 is None:
        raise NotImplementedError()

    rgb_sequence = frameseq.read_rgb_f32(frame_sequence_path)
    intrinsic_mat = to_opencv_camera_mat3x3(
        camera_parameters,
        rgb_sequence[0].shape[0]
    )

    def select_2d_points(id_, arr_frames_):
        arr_2d_points_ = []
        for fr in arr_frames_:
            # point_ = corner_storage[fr].points[corner_storage[fr].ids == id_][0]
            index = np.where(corner_storage[fr].ids == id_)[0][0]
            point_ = corner_storage[fr].points[index, :]
            arr_2d_points_.append(np.append(point_, np.array([1])))
        return np.array(arr_2d_points_)

    def ids_n_correspondences(arr_frames_):
        ids_ = []
        for i_ in arr_frames_:
            if i_ < 0 or i_ >= len(corner_storage):
                return False, None
            ids_.append(corner_storage[i_].ids)

        intersection_ids_ = np.array(ids_[0])
        for i_ in ids_[1:]:
            intersection_ids_ = np.intersect1d(intersection_ids_, i_)
        return True, intersection_ids_

    def pnp_by_frame_number(points_3d,
                            next_scene: int,
                            info: bool,
                            tvec_prev=np.empty((3, 1)),
                            rvec_prev=np.empty((3, 1)),
                            useExtrinsicGuess=False,
                            rep_error=2):
        ids_3d = None
        if len(points_3d) == 3:
            ids_3d = points_3d[1][points_3d[2] == True]  # id точек, для которых необходимо решать задачу pnp
        else:
            ids_3d = points_3d[1]

        ids_2d = corner_storage[next_scene].ids
        intersect_ids = np.intersect1d(ids_3d, ids_2d)  # пересекли по id 3d и 2d точки
        mask_3d = np.in1d(ids_3d, intersect_ids)
        mask_2d = np.in1d(ids_2d, intersect_ids)

        if points_3d[0][points_3d[2] == True][mask_3d].shape[0] < 5:
            print("мало точек для решения PnP")  # если у нас совсем мало точек, надо что-то делать
            return False, None, None, None

        retval_, r_vec_, t_vec_, inliers_ = cv2.solvePnPRansac(points_3d[0][points_3d[2] == True][mask_3d],
                                                               corner_storage[next_scene].points[mask_2d],
                                                               intrinsic_mat,
                                                               np.array([]),
                                                               reprojectionError=rep_error,
                                                               useExtrinsicGuess=useExtrinsicGuess,
                                                               rvec=rvec_prev,
                                                               tvec=tvec_prev,
                                                               iterationsCount=100,
                                                               confidence=0.99)
        if info and retval_:
            print("Кадр: ", next_scene, " количество соответствий: ", mask_2d[mask_2d].shape[0])

        if info and not retval_:
            print("Для кадра", next_scene, "не получилось решить решить задачу PnP с ошибкой репроекции:", rep_error,
                  "пикселей/я")

        ids_outliers_ = None
        if retval_:
            inliers_list_ = list(inliers_[:, 0])
            ids_outliers_ = []
            for j_ in range(points_3d[0][points_3d[2] == True][mask_3d].shape[0]):
                if j_ not in inliers_list_:
                    ids_outliers_.append(points_3d[1][points_3d[2] == True][mask_3d][j_])
        else:
            ids_outliers_ = list(points_3d[1])
        return retval_, r_vec_, t_vec_, ids_outliers_

    # давайте сначала получим облако 3d точек
    correspondences = build_correspondences(corner_storage[known_view_1[0]],
                                            corner_storage[known_view_2[0]])  # посмотрели на соответствия 2d точек
    new_points_3d, ids, median_cos = triangulate_correspondences(correspondences,
                                                                 pose_to_view_mat3x4(known_view_1[1]),
                                                                 pose_to_view_mat3x4(known_view_2[1]),
                                                                 intrinsic_mat,
                                                                 TriangulationParameters(20, 0, 0)
                                                                 )  # сделали триангуляцию соответствия

    storage_points_3d = [new_points_3d, ids, np.array([True] * ids.shape[0])]  # тут храним все полученные 3d точки
    print("Облако 3d точек - ", len(storage_points_3d[0]), "на кадрах:", min(known_view_1[0], known_view_2[0]),
          max(known_view_1[0], known_view_2[0]))

    # Достроим облако точек на кадре посредине
    min_ = min(known_view_1[0], known_view_2[0])
    max_ = max(known_view_1[0], known_view_2[0])
    middle_ = (known_view_1[0] + known_view_2[0]) // 2

    def solve_somewhere(left, right, left_view_mat, right_view_mat):
        correspondences = build_correspondences(corner_storage[left], corner_storage[right])
        print("пытаюсь добавить 3d точки на кадрах:", left, right)
        new_points_3d, ids, median_cos = triangulate_correspondences(correspondences,
                                                                     left_view_mat,
                                                                     right_view_mat,
                                                                     intrinsic_mat,
                                                                     TriangulationParameters(5, 0, 0)
                                                                     )
        if len(ids) != 0:
            new_3d_points_mask = np.array([True if i not in storage_points_3d[1] else False for i in ids])
            if new_points_3d[new_3d_points_mask].shape[0] != 0:
                storage_points_3d[0] = np.vstack((storage_points_3d[0], new_points_3d[new_3d_points_mask]))
                storage_points_3d[1] = np.hstack((storage_points_3d[1], ids[new_3d_points_mask]))
                storage_points_3d[2] = np.array([True] * storage_points_3d[0].shape[0])
                print("Облако 3d точек увеличилось до -", len(storage_points_3d[0]))

    arr_bi = []
    retval, r_vec, t_vec, ids_outliers = pnp_by_frame_number(storage_points_3d, middle_, False)
    if retval:
        view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
        arr_bi.append((middle_, max_, view_mat, pose_to_view_mat3x4(known_view_2[1])))
        # arr_bi.append((min_, middle_, pose_to_view_mat3x4(known_view_1[1]), view_mat))
    if min_ > 5:
        retval, r_vec, t_vec, ids_outliers = pnp_by_frame_number(storage_points_3d, min_ // 2, False)
        if retval:
            view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec)
            arr_bi.append((min_ // 2, min_, view_mat, pose_to_view_mat3x4(known_view_1[1])))

    for params in arr_bi:
        solve_somewhere(*params)

    # будем идти по shift кадров и добавлять 3d точки
    default_shift = 50  # abs(known_view_1[0] - known_view_2[0]) // 2
    shift = default_shift
    # идём вправо
    if max(known_view_1[0], known_view_2[0]) + shift < len(corner_storage):
        right_frame_number = max(known_view_1[0], known_view_2[0])
        prev_view = known_view_1[1] if known_view_1[0] == right_frame_number else known_view_2[1]
        prev_view_mat = pose_to_view_mat3x4(prev_view)
        # for i in range(right_frame_number + shift, len(corner_storage), shift):
        i = right_frame_number + shift
        while i < len(corner_storage):
            print("пытаюсь добавить 3d точки на кадрах:", right_frame_number, i)
            # 1 - определяем позицию камеры на этом кадре
            # 2 - доставляем 3d точки, которые можем доставить
            retval, r_vec, t_vec, ids_outliers = pnp_by_frame_number(storage_points_3d, i, False)
            if not retval:  # пробуем перезапуститься с другим сдвигом
                i -= shift
                shift = shift // 2
                i += shift
                if shift < 4:
                    print(
                        "Не удалось найти соседний кадр для увеличения успешного решения задачи PnP для увеличения облака точек")
                    shift = default_shift
                    right_frame_number += shift
                    i = right_frame_number
                    i += shift
                continue

            next_view_mat = rodrigues_and_translation_to_view_mat3x4(r_vec,
                                                                     t_vec)  # определили позицию камеры на данном кадре
            correspondences = build_correspondences(corner_storage[right_frame_number], corner_storage[i])
            new_points_3d, ids, median_cos = triangulate_correspondences(correspondences,
                                                                         prev_view_mat,
                                                                         next_view_mat,
                                                                         intrinsic_mat,
                                                                         TriangulationParameters(5, 0, 0)
                                                                         )

            # добавим только новые 3d точки
            if len(ids) != 0:
                new_3d_points_mask = np.array([True if i not in storage_points_3d[1] else False for i in ids])
                if new_points_3d[new_3d_points_mask].shape[0] != 0:
                    storage_points_3d[0] = np.vstack((storage_points_3d[0], new_points_3d[new_3d_points_mask]))
                    storage_points_3d[1] = np.hstack((storage_points_3d[1], ids[new_3d_points_mask]))
                    storage_points_3d[2] = np.array([True] * storage_points_3d[0].shape[0])
                    print("Облако 3d точек увеличилось до -", len(storage_points_3d[0]))
            prev_view_mat = next_view_mat

            if retval and shift != default_shift:  # если мы пробовали перезапускаться с меньшим сдвигом, то пробуем вернуться к изначальному шагу
                i = right_frame_number
                shift = default_shift
                i += shift

            right_frame_number += shift
            i += shift

    # в итоге после этих операций у нас есть облако 3d точек, для которых можно решать pnp для каждого кадра
    # давайте сделаем ретриангуляцию
    def retriangle(arr_frames):
        is_inside_frames, intersection_ids = ids_n_correspondences(arr_frames)
        if is_inside_frames:  # выберем нужные нам матрицы и 3d точки
            mats = []
            for i_retr in arr_frames:
                retval, r_vec, t_vec, ids_outliers = pnp_by_frame_number(storage_points_3d, i_retr,
                                                                                         False)
                if not retval:
                    print("Не получилось решить PnP в методе ретриангуляции")
                mats.append(intrinsic_mat @ rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec))
            for id_2d in intersection_ids:
                points_2d = select_2d_points(id_2d, arr_frames)
                point_3d = triangulate_nviews(mats, points_2d)
                el_number = np.where(id_2d == storage_points_3d[1])[0]
                storage_points_3d[0][el_number] = np.array(point_3d[0:3])
        else:  # надо что-то делать
            print("не те кадры для ретриангуляции")


    # storage_points_3d_old = storage_points_3d
    # min_retr = min_
    # while min_retr + 4 * default_shift < len(corner_storage):
    #     arr_frames_ = [min_retr + i * default_shift for i in range(4)]
    #     retriangle(arr_frames_)
    #     min_retr += default_shift
    # assert(storage_points_3d_old == storage_points_3d) # хочу проверить, что ретриангуляция вообще что-то поменяла

    # закончили с ретриангуляцией, пробуем найти итоговые положения
    view_mats = []
    # t_vec_prev = None
    # r_vec_prev = None
    r_vec_prev = view_mat3x4_to_rodrigues_and_translation(pose_to_view_mat3x4(known_view_1[1]))[0]
    t_vec_prev = view_mat3x4_to_rodrigues_and_translation(pose_to_view_mat3x4(known_view_1[1]))[1]
    storage_points_3d[2] = np.array([True] * storage_points_3d[0].shape[0])
    standart_repr_error = 2
    max_repr_error = 9
    rep_error = standart_repr_error
    # for i in range(len(corner_storage)):
    i = 0
    while i < len(corner_storage):
        retval, r_vec, t_vec, ids_outliers = pnp_by_frame_number(storage_points_3d, i, True,
                                                                 t_vec_prev, r_vec_prev, True,
                                                                 rep_error)
        if not retval:  # or ids_outliers.size < 10 если не получается пробуем доделать хоть как-то
            rep_error += 1
            if rep_error > max_repr_error:
                print("Не удалось решить PnP для поиска итогового положения камеры в кадре", i)
                break
            continue

        for j in ids_outliers:
            storage_points_3d[2][np.argwhere(storage_points_3d[1] == j)[0, 0]] = False
        t_vec_prev, r_vec_prev = t_vec, r_vec
        view_mats.append(rodrigues_and_translation_to_view_mat3x4(r_vec, t_vec))
        if rep_error > standart_repr_error:  # если заработало, пробуем опять с жёсткими ограничениями
            rep_error = standart_repr_error
        i += 1
    assert (len(corner_storage) == len(view_mats))

    point_cloud_builder = PointCloudBuilder(storage_points_3d[1],  # id всех найденных 3d точек
                                            storage_points_3d[0])

    calc_point_cloud_colors(
        point_cloud_builder,
        rgb_sequence,
        view_mats,
        intrinsic_mat,
        corner_storage,
        5.0
    )
    point_cloud = point_cloud_builder.build_point_cloud()
    poses = list(map(view_mat3x4_to_pose, view_mats))
    return poses, point_cloud


if __name__ == '__main__':
    # pylint:disable=no-value-for-parameter
    create_cli(track_and_calc_colors)()
