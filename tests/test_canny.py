import numpy as np

from src.ip.canny import canny_edge_detector, double_threshold_hysteresis, non_max_suppression


def test_nms_suppresses_non_maximum_neighbours():
    # a magnitude ridge 3 pixels wide, gradient pointing horizontally
    # (direction 0), so NMS should thin it to the single center column.
    magnitude = np.zeros((5, 7))
    magnitude[:, 2:5] = [1.0, 3.0, 1.0]
    direction = np.zeros((5, 7))  # all pointing along x -> "horizontal" sector
    out = non_max_suppression(magnitude, direction)
    assert np.all(out[:, 3] == 3.0)
    assert np.all(out[:, 2] == 0.0)
    assert np.all(out[:, 4] == 0.0)


def test_hysteresis_keeps_weak_pixel_connected_to_strong():
    mag = np.array([
        [0.0, 0.0, 0.0],
        [0.3, 0.9, 0.0],
        [0.0, 0.0, 0.0],
    ])
    edges = double_threshold_hysteresis(mag, low=0.2, high=0.5)
    assert edges[1, 0]  # weak pixel, connected to the strong pixel
    assert edges[1, 1]  # strong pixel


def test_hysteresis_drops_isolated_weak_pixel():
    mag = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.3, 0.0],
        [0.0, 0.0, 0.0],
    ])
    edges = double_threshold_hysteresis(mag, low=0.2, high=0.5)
    assert not edges.any()  # weak pixel with no strong neighbour anywhere


def test_canny_returns_boolean_edge_map_of_correct_shape():
    rng = np.random.default_rng(11)
    img = rng.random((40, 40)) * 255
    edges = canny_edge_detector(img)
    assert edges.shape == img.shape
    assert edges.dtype == bool


def test_canny_detects_a_clean_step_edge():
    img = np.zeros((30, 30))
    img[:, 15:] = 255.0
    edges = canny_edge_detector(img, low_threshold=0.1, high_threshold=0.2)
    # there should be edge pixels near column 15, away from the border
    assert edges[10:20, 13:18].any()


def test_canny_blank_image_has_no_edges():
    img = np.full((20, 20), 100.0)
    edges = canny_edge_detector(img)
    assert not edges.any()
