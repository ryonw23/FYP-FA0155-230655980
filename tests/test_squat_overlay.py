from powerlifting_coach.annotate_pose_video import draw_landmarks_on_bgr
from powerlifting_coach.motion_analysis import SQUAT_OVERLAY_CONNECTIONS


class _Frame:
    shape = (100, 100, 3)

    def copy(self):
        return self


class _Cv2Recorder:
    def __init__(self):
        self.lines = []
        self.circles = []

    def line(self, _frame, start, end, _colour, _width):
        self.lines.append((start, end))

    def circle(self, _frame, point, _radius, _colour, _fill):
        self.circles.append(point)


def test_squat_overlay_renders_both_sides_and_cross_body_connections_once():
    joint_y = {
        "shoulder": 0.1,
        "hip": 0.3,
        "knee": 0.5,
        "ankle": 0.7,
        "foot_index": 0.9,
    }
    landmarks = {
        f"{side}_{joint}": {
            "x": x,
            "y": joint_y[joint],
            "visibility": 1.0,
            "presence": 1.0,
        }
        for side, x in (("left", 0.25), ("right", 0.75))
        for joint in ("shoulder", "hip", "knee", "ankle", "foot_index")
    }
    connections = SQUAT_OVERLAY_CONNECTIONS
    displayed_landmarks = {name for connection in connections for name in connection}
    cv2 = _Cv2Recorder()

    draw_landmarks_on_bgr(
        cv2,
        _Frame(),
        landmarks,
        connections,
        displayed_landmarks=displayed_landmarks,
    )

    left_chain = (((25, 10), (25, 30)), ((25, 30), (25, 50)),
                  ((25, 50), (25, 70)), ((25, 70), (25, 90)))
    right_chain = (((75, 10), (75, 30)), ((75, 30), (75, 50)),
                   ((75, 50), (75, 70)), ((75, 70), (75, 90)))

    assert all(connection in cv2.lines for connection in left_chain)
    assert all(connection in cv2.lines for connection in right_chain)
    assert cv2.lines.count(((25, 10), (75, 10))) == 1
    assert cv2.lines.count(((25, 30), (75, 30))) == 1
    assert set(cv2.circles) == {
        (x, y)
        for x in (25, 75)
        for y in (10, 30, 50, 70, 90)
    }
    assert len(cv2.circles) == 10
