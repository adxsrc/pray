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

    def geode(self, L=None):
        self.geode = Geode(self.metric, 0, None, L=L, **self.kwargs)
