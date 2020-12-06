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


from fadge.metric import KerrSchild
from jax          import numpy as np
from jax.experimental.maps import xmap

from .geode       import Geode
from .integrator  import RK4

class PRay:

    def __init__(self, aspin=0):
        axmap = {0:'alpha', 1:'beta'}
        rhs   = Geode(KerrSchild(aspin))
        self.rhs    = xmap(rhs, in_axes=axmap, out_axes=axmap)
        self.step   = RK4
        self.t      = 0
        self.states = [np.array([[
            [0, 3, 3, 0, 1, -1, 0, 0],
            [0, 3, 3, 0, 1, -1, 0, 0],
        ]], dtype=np.float32)]


    def integrate(self, tlist):
        for t in tlist:
            if t != self.t:
                self.states.append(self.step(self.rhs, self.states[-1], t - self.t))
                self.t = t
