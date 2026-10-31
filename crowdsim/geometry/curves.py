"""Closed convex base curves and their parallel (offset) curves.

A stadium bowl is described in "bowl coordinates" (s, d):

* ``s`` in [0, 1) is the position around the field boundary, as a fraction
  of the boundary's arc length (s = 0 is on the +x axis, increasing
  anticlockwise).
* ``d`` >= 0 is the outward distance from the field boundary.

The point at (s, d) is P(s) + d * N(s), where N is the outward unit normal.
Offset curves of a convex curve share its normals, so a line of constant s
is a straight radial line (used for aisles, vomitories, stairs and exits).
For a convex closed curve the arc length at offset d obeys

    ell_d(s) = ell_0(s) + d * phi(s),        total length = L_0 + 2*pi*d,

where phi(s) is the accumulated turning angle of the tangent. This is used
for exact seat spacing along each row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * np.pi


@dataclass(frozen=True, eq=False)
class RingCurve:
    """A closed, convex, anticlockwise curve resampled uniformly by arc length."""

    points: np.ndarray   # (M, 2) base-curve samples
    normals: np.ndarray  # (M, 2) outward unit normals
    ell0: np.ndarray     # (M+1,) cumulative arc length, ell0[-1] = L0
    phi: np.ndarray      # (M+1,) cumulative turning angle, phi[-1] = 2*pi

    # ------------------------------------------------------------ builders
    @classmethod
    def superellipse(cls, a: float, b: float, n: float, samples: int = 4096) -> "RingCurve":
        """|x/a|^n + |y/b|^n = 1.  n = 2 ellipse, n -> large rounded rectangle."""
        if a <= 0 or b <= 0:
            raise ValueError("superellipse semi-axes a and b must be > 0")
        if n < 2:
            raise ValueError("superellipse exponent n must be >= 2 (convex shape)")
        t = np.linspace(0.0, TWO_PI, samples * 50, endpoint=False)
        c, s = np.cos(t), np.sin(t)
        x = a * np.sign(c) * np.abs(c) ** (2.0 / n)
        y = b * np.sign(s) * np.abs(s) ** (2.0 / n)
        return cls.from_polyline(np.column_stack([x, y]), samples)

    @classmethod
    def from_polyline(cls, dense: np.ndarray, samples: int) -> "RingCurve":
        dense = np.asarray(dense, dtype=float)
        closed = np.vstack([dense, dense[:1]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        cum = np.concatenate([[0.0], np.cumsum(seg)])
        target = np.linspace(0.0, cum[-1], samples, endpoint=False)
        pts = np.column_stack([np.interp(target, cum, closed[:, 0]),
                               np.interp(target, cum, closed[:, 1])])
        # enforce anticlockwise orientation, keeping pts[0] as the start point
        x, y = pts[:, 0], pts[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        if area < 0:
            pts = np.vstack([pts[:1], pts[:0:-1]])
        tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        normals = np.column_stack([tang[:, 1], -tang[:, 0]])  # outward for CCW
        closed_pts = np.vstack([pts, pts[:1]])
        ell0 = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed_pts, axis=0), axis=1))])
        ang = np.arctan2(tang[:, 1], tang[:, 0])
        ang = np.unwrap(np.concatenate([ang, ang[:1]]))
        phi = ang - ang[0]
        phi[-1] = TWO_PI
        phi = np.clip(np.maximum.accumulate(phi), 0.0, TWO_PI)  # convex => monotone
        return cls(pts, normals, ell0, phi)

    # ------------------------------------------------------------- queries
    @property
    def m(self) -> int:
        return len(self.points)

    @property
    def length(self) -> float:
        """Perimeter of the base (field) curve, L0."""
        return float(self.ell0[-1])

    @property
    def s_grid(self) -> np.ndarray:
        return np.arange(self.m + 1) / self.m

    def ring_length(self, d: float) -> float:
        """Perimeter of the offset curve at distance d (= L0 + 2*pi*d)."""
        return float(self.ell0[-1] + d * self.phi[-1])

    def xy(self, s, d) -> np.ndarray:
        """Cartesian position(s) of bowl coordinates (s, d). Broadcasts."""
        s = np.asarray(s, dtype=float) % 1.0
        d = np.asarray(d, dtype=float)
        u = s * self.m
        i = np.floor(u).astype(int) % self.m
        f = (u - np.floor(u))[..., None]
        j = (i + 1) % self.m
        p = (1.0 - f) * self.points[i] + f * self.points[j]
        nrm = (1.0 - f) * self.normals[i] + f * self.normals[j]
        nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)
        return p + d[..., None] * nrm

    def arc(self, s, d) -> np.ndarray:
        """Arc-length coordinate of (s, d) measured along the offset curve from s = 0."""
        s = np.asarray(s, dtype=float) % 1.0
        grid = self.s_grid
        return np.interp(s, grid, self.ell0) + np.asarray(d, dtype=float) * np.interp(s, grid, self.phi)

    def s_at_arc(self, ell, d: float) -> np.ndarray:
        """Inverse of :meth:`arc` for a single offset d."""
        g = self.ell0 + d * self.phi
        ell = np.asarray(ell, dtype=float) % g[-1]
        return np.interp(ell, g, self.s_grid)

    def arc_between(self, s0: float, s1: float, d: float) -> float:
        """Forward (anticlockwise) arc length from s0 to s1 along the offset curve at d."""
        a0, a1 = float(self.arc(s0, d)), float(self.arc(s1, d))
        return a1 - a0 if a1 >= a0 else self.ring_length(d) - a0 + a1

    def polyline(self, d: float, closed: bool = True) -> np.ndarray:
        pts = self.points + d * self.normals
        return np.vstack([pts, pts[:1]]) if closed else pts
