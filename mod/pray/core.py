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


from .icond import cam

from fadge.metric import KerrSchild
from fadge.geode  import Geode
from fadge.utils  import Nullify

from xaj import DP5

from jax import numpy as np
from jax.experimental.maps import xmap


class PRay:

    def __init__(self, aspin=0, r_obs=1000, i_obs=60, j_obs=0, *args, **kwargs):
        metric  = KerrSchild(aspin)
        geode   = Geode(metric)
        nullify = Nullify(metric)

        rij = np.array([r_obs, np.radians(i_obs), np.radians(j_obs)])
        def icond(ab): # closure on rij
            s = cam(rij, ab)
            return np.concatenate([s[:4], nullify(s[:4], s[4:])])

        smap = {0:'alpha', 1:'beta'}
        a, b = np.linspace(-10,10,65), np.linspace(-10,10,65)
        ab   = np.array(np.meshgrid(a, b)).T

        rhs   = xmap(geode, in_axes=smap, out_axes=smap)
        state = xmap(icond, in_axes=smap, out_axes=smap)(ab)

        self.sol = DP5(lambda l, s: rhs(s), 0.0, state, *args, **kwargs)

    def __call__(self, *args, **kwargs):
        self.sol(*args, **kwargs)

    @property
    def lambdas(self):
        return self.sol.xs

    @property
    def states(self):
        return self.sol.ys
