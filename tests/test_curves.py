import math

import numpy as np
import pytest

from crowdsim.geometry import RingCurve


@pytest.fixture(scope="module")
def circle():
    return RingCurve.superellipse(10.0, 10.0, 2.0)


def test_circle_perimeter(circle):
    assert circle.length == pytest.approx(2 * math.pi * 10.0, rel=1e-5)


def test_offset_perimeter_formula(circle):
    # convex curve: perimeter at offset d = L0 + 2*pi*d, check against the polyline itself
    for d in (0.0, 5.0, 30.0):
        poly = circle.polyline(d)
        numeric = np.sum(np.linalg.norm(np.diff(poly, axis=0), axis=1))
        assert circle.ring_length(d) == pytest.approx(numeric, rel=1e-5)


def test_xy_is_on_offset_curve(circle):
    s = np.linspace(0, 1, 37, endpoint=False)
    p = circle.xy(s, 4.0)
    assert np.allclose(np.linalg.norm(p, axis=1), 14.0, atol=1e-3)


def test_s_at_arc_inverts_arc(circle):
    s = np.array([0.0, 0.1, 0.5, 0.9])
    for d in (0.0, 7.5):
        assert np.allclose(circle.s_at_arc(circle.arc(s, d), d), s, atol=1e-9)


def test_arc_between_wraps(circle):
    d = 2.0
    total = circle.ring_length(d)
    assert circle.arc_between(0.25, 0.75, d) == pytest.approx(total / 2, rel=1e-4)
    assert circle.arc_between(0.75, 0.25, d) == pytest.approx(total / 2, rel=1e-4)


def test_superellipse_rejects_concave():
    with pytest.raises(ValueError):
        RingCurve.superellipse(10, 5, 1.5)


def test_rounded_rectangle_is_between_ellipse_and_rectangle():
    a, b = 60.0, 42.0
    ell = RingCurve.superellipse(a, b, 2.0).length
    rr = RingCurve.superellipse(a, b, 4.0).length
    assert ell < rr < 4 * (a + b)
