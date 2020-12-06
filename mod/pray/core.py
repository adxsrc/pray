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


from .geode      import Geode
from .integrator import RK4
from .icond      import cam

from fadge.metric import KerrSchild
from fadge.utils  import Nullify

from jax import numpy as np
from jax.experimental.maps import xmap


class PRay:

    def __init__(self, aspin=0, r_obs=1000, i_obs=60, j_obs=0):
        metric  = KerrSchild(aspin)
        nullify = Nullify(metric)
        rhs     = Geode(metric)

        rij = np.array([r_obs, np.radians(i_obs), np.radians(j_obs)])
        def icond(ab):
            s = cam(rij, ab)
            return np.concatenate([s[:4], nullify(s[:4], s[4:])])

        axmap = {0:'alpha', 1:'beta'}
        a, b  = np.linspace(-10,10,65), np.linspace(-10,10,65)
        ab    = np.array(np.meshgrid(a, b)).T

        self.rhs    =  xmap(rhs,   in_axes=axmap, out_axes=axmap)
        self.states = [xmap(icond, in_axes=axmap, out_axes=axmap)(ab)]
        self.step   = RK4
        self.t      = 0

    def integrate(self, tlist):
        for t in tlist:
            if t != self.t:
                self.states.append(self.step(self.rhs, self.states[-1], t - self.t))
                self.t = t
