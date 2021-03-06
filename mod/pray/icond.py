# Copyright (C) 2020 Chi-kwan Chan
# Copyright (C) 2020 Steward Observatory
#
# This file is part of PRay.
#
# PRay is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PRay is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PRay.  If not, see <http://www.gnu.org/licenses/>.


from jax import numpy as np
from fadge.utils import Nullify


def cam(rij, ab):

    ci, si = np.cos(rij[1]), np.sin(rij[1])
    cj, sj = np.cos(rij[2]), np.sin(rij[2])

    R0 = rij[0] * si - ab[1] * ci # cylindrical radius
    z  = rij[0] * ci + ab[1] * si
    y  = R0     * sj + ab[0] * cj
    x  = R0     * cj - ab[0] * sj

    return np.array([
        [0, x, y, z],
        [1, si * cj, si * sj, ci],
    ], dtype=ab.dtype)


def sphorbit(aspin, r0):

    def PHI(a, r):
        if a == 0 and r == 3:
            return 0 # 2 * a + (9/2) * (r-3) / a
        elif a == 1:
            return - (r * r - 2 * r - 1)
        else:
            return - (r * r * r - 3 * r * r + a * a * r + a * a) / (a * (r - 1))

    def Q(a, r):
        if a == 0 and r == 3:
            return 27 # (r**3 / (r-1)**2) * (4 - r * (r-3)**2 / a**2)
        if a == 1:
            return - r*r*r * (r - 4)
        else:
            return - (r*r*r * (r*r*r - 6*r*r + 9*r - 4*a*a)) / (a*a * (r - 1) * (r - 1))

    def thetadot(a, r):
        return np.sqrt(Q(a, r))

    def phidot(a, r):
        return (2*r*a + (r*r - 2*r) * PHI(a, r)) / (r*r + a*a - 2*r)

    R = np.sqrt(r0*r0 + aspin*aspin)
    return np.array([
        [0, R, 0, 0],
        [1, 0, R * phidot(aspin, r0), -r0 * thetadot(aspin, r0)],
    ])
