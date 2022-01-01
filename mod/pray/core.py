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

from jax import numpy as np
from jax.experimental.maps import xmap


class PRay(Geode):

    def __init__(self,
        aspin=0,
        **kwargs,
    ):
        self.metric  = KerrSchild(aspin)
        self.nullify = Nullify(self.metric)
        self.kwargs  = kwargs

    def set_cam(self, r_obs=1e4, i_obs=60, j_obs=0):
        self.rij = np.array([r_obs, np.radians(i_obs), np.radians(j_obs)])

    def set_pixels(self, a, b):
        def ic(ab): # closure on self.rij and self.nullify
            s = cam(self.rij, ab)
            return np.array([s[0], self.nullify(s[0],s[1])])
        ab = np.array([a, b])
        self.s0 = xmap(
            ic,
            in_axes ={i  :i for i in range(1,ab.ndim)},
            out_axes={i+1:i for i in range(1,ab.ndim)},
        )(ab)

    def geode(self, L=None):
        self.geode = Geode(self.metric, 0, self.s0, L=L, **self.kwargs)
