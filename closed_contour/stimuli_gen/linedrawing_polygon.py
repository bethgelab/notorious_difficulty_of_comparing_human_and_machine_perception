# This code generates images of size 256 x 256 px that contains either an open or closed contour.
# This contour consists of straight lines that form polygons.
# author: Christina Funke

import numpy as np
import math
import csv
import os
from PIL import Image, ImageDraw
from pathlib import Path

# -------------------------------------------------------
# code to determine overlap
# -------------------------------------------------------

# that's the distance of two non-overlaping points (=10 px)
dist = 40

# intersection
def ccw(A, B, C):
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def test_overlap_point(lines, point, dist):
    # point=(0,0)
    # lines=[(1,1),(2,2),(3,3)]
    return point_to_multipleline_dist(lines, point) < dist


def test_overlap_line(lines, new_line, dist):
    # new_line=[(10,10),(20,20),(30,30)]
    # lines=[(1,1),(2,2),(3,3)]
    # overlap
    for i in range(len(new_line)):
        point = new_line[i : i + 1][0]
        if test_overlap_point(lines, point, dist):
            return True
    for i in range(len(lines)):
        point = lines[i : i + 1][0]
        if test_overlap_point(new_line, point, dist):
            return True

    # intersection
    for i in range(len(new_line) - 1):
        line1 = new_line[i : i + 2]
        A = np.array(line1[0])
        B = np.array(line1[1])
        for i in range(len(lines) - 1):
            line2 = lines[i : i + 2]
            C = np.array(line2[0])
            D = np.array(line2[1])
            if intersect(A, B, C, D):
                return True
    return False


def test_overlap_all_lines(all_lines, new_line, dist):
    # new_line=[(10,10),(20,20),(30,30)]
    # all_lines=[[(1,1),(2,2),(3,3)],[(10,10),(20,20),(30,30)]]
    for lines in all_lines:
        if test_overlap_line(lines, new_line, dist):
            return True
    return False


def point_to_multipleline_dist(lines, point):
    min_dist = +np.inf
    for i in range(len(lines) - 1):
        line = lines[i : i + 2]
        min_dist = min(min_dist, point_to_line_dist(line, point))
    return min_dist


def point_to_line_dist(line, point):
    x1, y1 = line[0]
    x2, y2 = line[1]
    x3, y3 = point
    px = x2 - x1
    py = y2 - y1

    something = px * px + py * py
    u = ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py
    dx = x - x3
    dy = y - y3

    dist = math.sqrt(dx * dx + dy * dy)
    return dist


# -------------------------------------------------------
# code to make contours and flanker
# -------------------------------------------------------


def make_contour(polygon_points):
    """
    method 1
    """
    # position vertices on circle and add noise
    r = np.random.uniform(128, 256)
    noise = r / 3
    points = []
    for n in range(polygon_points):
        phi = 2 * np.pi / polygon_points * n
        x = r * np.cos(phi)
        y = r * np.sin(phi)
        x += np.random.random() * noise * 2 - noise
        y += np.random.random() * noise * 2 - noise
        points.append((x, y))

    points.append(points[0])
    return points


def get_random_angles(polygon_points):
    a = np.random.random(polygon_points)
    a /= a.sum()
    return np.cumsum(a * 2 * np.pi)


def make_flanker(r, r2, pos, number_of_segments):
    points = []
    angle = np.random.rand() * 2 * np.pi
    points.append((np.sin(angle) * r + pos[0], np.cos(angle) * r + pos[1]))

    points.append(pos)
    if number_of_segments == 2:
        angle2 = angle + np.random.rand() * 3 / 4 * np.pi + 1 / 4 * np.pi
        points.append((np.sin(angle2) * r2 + pos[0], np.cos(angle2) * r2 + pos[1]))
    return points


def move_inside(img_size, points, width):
    if test_inside(img_size, points, width):
        return True, points
    for n in range(50):
        pos = [np.random.uniform(0, img_size), np.random.uniform(0, img_size)]
        points = shift_to_pos(points, pos)
        if test_inside(img_size, points, width):
            return True, points
    return False, points


def test_inside(img_size, points, width):
    margin = 20
    for point in points:
        if (
            point[0] < width / 2 + margin
            or point[0] > img_size - width / 2 - margin
            or point[1] < width / 2 + margin
            or point[1] > img_size - width / 2 - margin
        ):
            return False
    return True


def test_visible(img_size, points):
    for point in points:
        if img_size > point[0] > 0 and img_size > point[1] > 0:
            return True
    return False


# -------------------------------------------------------
# code to transform closed to open contour
# -------------------------------------------------------


def remove_one_line(line):
    """
    method a
    """
    start = np.random.randint(0, len(line) - 1)
    line = line[start:-1] + line[0:start]

    return line, start


def dist_angle(a, b, c):
    # a, b, c are three points.
    # return angle at position b, r is the distance between a, b
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    f = b - a
    e = b - c

    f_u = f / np.linalg.norm(f)
    e_u = e / np.linalg.norm(e)
    ang2 = np.arctan2(f[0], f[1])
    ang1 = np.arctan2(e[0], e[1])

    angle = (ang1 - ang2) % (2 * np.pi)
    r = np.linalg.norm(f)
    return angle, r


def shuffle_line(line):
    """
    method b
    """
    # get angle and radii
    angles = []
    radii = []
    line.append(line[1])
    for i in range(len(line) - 2):
        angle, radius = dist_angle(line[i], line[i + 1], line[i + 2])
        angles.append(angle)
        radii.append(radius)

    # shuffle
    np.random.shuffle(angles)
    np.random.shuffle(radii)

    # reconstruct
    new_a = [(512, 512)]

    overlap = False
    for i in range(len(angles)):
        if i == 0:
            old_angle = np.random.rand() * 2 * np.pi
        else:
            x0_diff = new_a[-1][0] - new_a[-2][0]
            y0_diff = new_a[-1][1] - new_a[-2][1]
            old_angle = np.arctan2(x0_diff, y0_diff)

        angle = old_angle - (np.pi - angles[i])
        r = radii[i]

        new_x_diff = np.sin(angle) * r
        new_y_diff = np.cos(angle) * r
        new_point = (new_a[-1][0] + new_x_diff, new_a[-1][1] + new_y_diff)

        new_a.append(new_point)

        overlap = max(overlap, test_overlap_line(new_a[0:-2], new_a[-2:], dist=dist))

    return new_a, overlap


def make_contour_both(polygon_points):
    """
    method c
    """

    diff = np.random.rand() * 15 * 4 + 10 * 4  # differenz zwischen 20px und 50px
    minr = 0
    radius = 256 - minr

    while 1:

        radii = np.random.random(polygon_points) * radius + minr
        radii_closed = np.copy(radii)
        phis = get_random_angles(polygon_points)

        # add point
        phis = np.append(phis, phis[0])
        radii = np.append(radii, radii[0])

        s = np.random.randint(0, 2)
        radii[0] += diff * (-1) ** s
        radii[-1] -= diff * (-1) ** s

        ov = False
        start_angle = np.random.rand() * 2 * np.pi

        points = []
        for n in range(polygon_points + 1):
            r = radii[n]
            phi = phis[n] + start_angle
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            points.append((x, y))
            ov = (
                ov
                or test_overlap_line(points[0:-2], points[-2:], dist)
                or test_overlap_point(points[0:-1], points[-1], dist)
            )  # teste die punkte

        ov = ov or test_overlap_point(
            points[1:], points[0], dist
        )  # teste den ersten punkt

        points_closed = []
        for n in range(polygon_points):
            r = radii_closed[n]
            phi = phis[n] + start_angle
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            points_closed.append((x, y))
            ov = (
                ov
                or test_overlap_line(points_closed[0:-2], points_closed[-2:], dist)
                or test_overlap_point(points[0:-1], points[-1], dist)
            )

        points_closed.append(points_closed[0])
        ov = (
            ov
            or test_overlap_line(points_closed[1:-2], points_closed[-2:], dist)
            or test_overlap_point(points_closed[0:-1], points[-1], dist)
        )

        ov = ov or test_overlap_point(points[1:], points[0], dist)

        # if polygon_points==3:
        #    ov=ov or (test_overlap_point([points_closed[2],points_closed[1]], points_closed[0], dist) or
        #        test_overlap_point([points_closed[3],points_closed[2]], points_closed[1], dist) or
        #        test_overlap_point([points_closed[1],points_closed[3]], points_closed[2], dist))
        if not ov:
            break

    return points, points_closed


def shift_to_pos(new_a, pos):
    x = np.array(new_a)[:, 0]
    y = np.array(new_a)[:, 1]
    pos_x = x.mean()
    pos_y = y.mean()
    x -= pos_x - pos[0]
    y -= pos_y - pos[1]

    new_a_m = []
    for i in range(len(new_a)):
        new_a_m.append((x[i], y[i]))
    return new_a_m


def shift_to_pos2(new_a, ref_a, set_num, anchor=2):
    x = np.array(new_a)[:, 0]
    y = np.array(new_a)[:, 1]
    if set_num == 9 or set_num == 7:
        x_diff = new_a[0][0] - ref_a[anchor][0]
        y_diff = new_a[0][1] - ref_a[anchor][1]
    else:
        x_diff = new_a[anchor][0] - ref_a[anchor][0]
        y_diff = new_a[anchor][1] - ref_a[anchor][1]

    x -= x_diff
    y -= y_diff

    new_a_m = []
    for i in range(len(new_a)):
        new_a_m.append((x[i], y[i]))
    return new_a_m


# -------------------------------------------------------
# define and draw image
# -------------------------------------------------------


def define_image(polygon_points, set_num):
    img_size = 1024
    width = 10
    # draw main contour
    if polygon_points == 0:
        open_lines = []
        closed_lines = []
        number_of_flankers = 25

    else:
        inside = False
        while not inside:

            points_contour_open, points_contour_closed = make_contour_both(
                polygon_points=polygon_points
            )
            if set_num == 9 or set_num == 7:
                points_contour_closed = make_contour(polygon_points=polygon_points)
                points_contour_open, anchor = remove_one_line(points_contour_closed)
                if set_num == 7:
                    points_contour_open = points_contour_open[0:-1]
            else:
                anchor = 2

            # shift to position
            pos = [np.random.uniform(0, img_size), np.random.uniform(0, img_size)]
            points_contour_closed = shift_to_pos(points_contour_closed, pos)
            points_contour_open = shift_to_pos2(
                points_contour_open,
                points_contour_closed,
                set_num=set_num,
                anchor=anchor,
            )

            inside = test_inside(img_size, points_contour_open, width) and test_inside(
                img_size, points_contour_closed, width
            )

        open_lines = []
        closed_lines = []
        open_lines.append(points_contour_open)
        closed_lines.append(points_contour_closed)
        if set_num == 6:
            number_of_flankers = 0
        else:
            number_of_flankers = np.random.randint(10, 26)

    # closed_lines_main is introduced to reduce computation time:
    # When checking for overlap (in the next for loop) closed_lines_main is used to avoid to loop over all flankers twice.
    closed_lines_main = np.copy(closed_lines)
    # draw flanker
    for flanker_num in range(number_of_flankers):
        number_of_segments = np.random.randint(1, 3)

        r = np.random.uniform(128, 256)
        if set_num == 24:
            r2 = np.random.uniform(32, 64)
            number_of_segments = 2
        else:
            r2 = np.copy(r)

        visible = False
        overlap = True
        while overlap or not visible:
            pos = (np.random.uniform(0, img_size), np.random.uniform(0, img_size))
            new_line = make_flanker(
                r=r, r2=r2, pos=pos, number_of_segments=number_of_segments
            )
            visible = test_visible(img_size, new_line)
            if (
                visible
            ):  # to save time, only test for overlap if visible (if not visible flanker will be resampled anyway)
                overlap = test_overlap_all_lines(
                    closed_lines_main, new_line, dist
                ) or test_overlap_all_lines(open_lines, new_line, dist)

            # inside = test_inside(img_size, new_line, width)

        open_lines.append(new_line)
        closed_lines.append(new_line)

    return open_lines, closed_lines


def draw_image(all_lines, set_num):
    img_size = 1024
    if set_num == 2:
        width = 5
    elif set_num == 3:
        width = 18
    elif set_num == 8:
        width = 30
    else:
        width = 10
    img = Image.new("RGB", [img_size, img_size], "white")
    draw = ImageDraw.Draw(img)

    c = 0
    for line in all_lines:
        c += 1

        num_lines = range(len(line) - 1)
        color = (0, 0, 0)

        for i in num_lines:
            draw.line(line[i] + line[i + 1], width=width, fill=color)
            for point in line:  # damit es keine Löcher an den Ecken gibt
                draw.ellipse(
                    (
                        point[0] - width / 2,
                        point[1] - width / 2,
                        point[0] + width / 2,
                        point[1] + width / 2,
                    ),
                    fill=color,
                )
    scale = 4
    img_anti = img.resize((img_size // scale, img_size // scale), Image.ANTIALIAS)

    return img_anti


def draw_image_bwb(all_lines):
    img_size = 1024
    width = 18
    widthw = 6
    img = Image.new("RGB", [img_size, img_size], "white")
    draw = ImageDraw.Draw(img)

    c = 0
    for line in all_lines:
        c += 1

        num_lines = range(len(line) - 1)

        color = (0, 0, 0)
        for i in num_lines:
            draw.line(line[i] + line[i + 1], width=width, fill=color)
            for point in line:  # damit es keine Löcher an den Ecken gibt
                draw.ellipse(
                    (
                        point[0] - width / 2,
                        point[1] - width / 2,
                        point[0] + width / 2,
                        point[1] + width / 2,
                    ),
                    fill=color,
                )

    c = 0
    for line in all_lines:
        c += 1

        num_lines = range(len(line) - 1)

        color = (255, 255, 255)
        for i in num_lines:
            draw.line(line[i] + line[i + 1], width=widthw, fill=color)
            for point in line:  # damit es keine Löcher an den Ecken gibt
                draw.ellipse(
                    (
                        point[0] - widthw / 2,
                        point[1] - widthw / 2,
                        point[0] + widthw / 2,
                        point[1] + widthw / 2,
                    ),
                    fill=color,
                )

    scale = 4
    img_anti = img.resize((img_size // scale, img_size // scale), Image.ANTIALIAS)

    return img_anti


def set_seed_rep(method):
    """
    Set seed and number of repetitions. Both depend on the type of the data set. For example a different seed is used
    for test and training set
    :param method: type of the data set [string]
    :return: number of [int]
    """
    if method.endswith("val"):
        np.random.seed(0)
        num_rep = 200
    elif method.endswith("test"):
        np.random.seed(1)
        num_rep = 400
    elif method.endswith("train"):
        np.random.seed(2)
        num_rep = 2000
    return num_rep


# -------------------------------------------------------
# main function that combines the calculations
# -------------------------------------------------------


def set1_otf(closedness):
    """
    This function can be used to generate a training image of set1 on-the-fly
    :param closedness: label, either 0 (closed) or 1 (open) [int]
    :return: line-drawing
    """
    polygon_points = np.random.randint(3, 10)
    open_lines, closed_lines = define_image(polygon_points, 1)
    if closedness == 0:
        return draw_image(closed_lines, 1)
    elif closedness == 1:
        return draw_image(open_lines, 1)


def make_full_dataset(top_dir, set_num, debug):
    """
    generate and save the full data set for a specified variation
    :param top_dir: where to save the images [string]
    :param set_num: number that specifies the variation [one of: 1-13, 24, 25]
    :param debug: generate only seven images [bool]
    """

    save = True
    stim_folder = top_dir + "set" + str(set_num) + "/linedrawing/"
    if set_num == 1:
        methods = ["val", "test", "train"] # remove "train" from list to not generate training set
    else:
        methods = ["test"]
    for method in methods:
        print(method)

        num_rep = set_seed_rep(method)
        if debug:
            num_rep = 1
        number = 0  # nummer des letzten vorhandenen bildes (0 sonst)

        # make folder
        new_folder = os.path.join(stim_folder, method)
        print(new_folder)
        if not Path(new_folder).is_dir():
            print("make new folder")
            os.makedirs(new_folder)

        if save:
            with open(os.path.join(stim_folder, method, method + ".csv"), "a") as f:
                writer = csv.writer(f)
                writer.writerow(["image_name", "points", "closed_contour"])

        # make folders: open, closed
        new_folder = os.path.join(stim_folder, method, "open")
        if not Path(new_folder).is_dir():
            os.mkdir(new_folder)

        new_folder = os.path.join(stim_folder, method, "closed")
        if not Path(new_folder).is_dir():
            os.mkdir(new_folder)

        for rep in range(num_rep):
            print(rep, end="", flush=True)
            if set_num == 10:
                polygon_pointss = [3, 3, 3, 3, 3, 3, 3]
            elif set_num == 11:
                polygon_pointss = [6, 6, 6, 6, 6, 6, 6]
            elif set_num == 12:
                polygon_pointss = [9, 9, 9, 9, 9, 9, 9]
            elif set_num == 13:
                polygon_pointss = [12, 12, 12, 12, 12, 12, 12]
            else:
                polygon_pointss = [3, 4, 5, 6, 7, 8, 9]
            for polygon_points in polygon_pointss:

                open_lines, closed_lines = define_image(polygon_points, set_num)
                for closed_contour in [True, False]:
                    number += 1
                    if closed_contour:
                        if set_num == 5:
                            res = draw_image_bwb(closed_lines)
                        else:
                            res = draw_image(closed_lines, set_num)
                    else:
                        if set_num == 5:
                            res = draw_image_bwb(open_lines)
                        else:
                            res = draw_image(open_lines, set_num)

                    if set_num == 25:
                        res = res.convert("1")
                    res = res.convert("RGB")
                    if save:
                        filename = method + str(number)
                        with open(
                            os.path.join(stim_folder, method, method + ".csv"), "a"
                        ) as f:
                            writer = csv.writer(f)
                            writer.writerow(
                                [filename, str(polygon_points), str(closed_contour)]
                            )

                        if closed_contour:
                            folder = "closed"
                        else:
                            folder = "open"
                        res.save(
                            os.path.join(stim_folder, method, folder, filename + ".png")
                        )