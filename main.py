import sys
from collections.abc import Iterable
from typing import Dict, List, Tuple

import gt4py
import netCDF4 as nc
import numpy as np
from gt4py.gtscript import IJ, IJK, Field, I, J, K, stencil

from riem_solver_c import riem_solver_c


def iterable(obj) -> bool:
    return isinstance(obj, Iterable)


class Dataset:
    def __init__(self, file_name: str, backend: str):
        self._dataset = nc.Dataset(file_name)
        self._backend = backend
        self.npx = int(self._dataset["npx"][0].item())
        self.npy = int(self._dataset["npy"][0].item())
        self.ng = int(self._dataset["ng"][0].item())
        self.km = int(self._dataset["km"][0].item())

    @staticmethod
    def _find_fuzzy(axes: List[str], name: str) -> int:
        return next(i for i, axis in enumerate(axes) if axis.startswith(name))

    def netcdf_to_gt4py(self, var: str) -> gt4py.storage.Storage:
        """Convert a netcdf variable to gt4py storage."""
        axes = [d.name for d in var.get_dims()]
        idim = self._find_fuzzy(axes, "xaxis")
        jdim = self._find_fuzzy(axes, "yaxis")
        kdim = self._find_fuzzy(axes, "zaxis")
        if np.prod(var.shape) > 1:
            permutation = [dim for dim in (idim, jdim, kdim) if dim]
            # put other axes at the back
            for i in range(len(axes)):
                if i not in permutation:
                    permutation.append(i)
            ndarray = np.squeeze(np.transpose(var, permutation))
            if len(ndarray.shape) == 3:
                origin = (self.ng, self.ng, 0)
            elif len(ndarray.shape) == 2:
                origin = (self.ng, self.ng)
            else:
                origin = (0,)
            return gt4py.storage.from_array(
                ndarray, backend, default_origin=origin, shape=ndarray.shape)
        else:
            return var[0].item()

    def __getitem__(self, index: str) -> gt4py.storage.Storage:
        variable = self._dataset[index]
        return self.netcdf_to_gt4py(variable)

    def new(self, axes: Tuple[gt4py.gtscript._Axis],
            dtype, pad_k=False) -> gt4py.storage.Storage:
        k_add = 1 if pad_k else 0
        if axes == IJK:
            origin = (self.ng, self.ng, 0)
            shape = (2 * self.ng + self.npx, 2 * self.ng + self.npy, self.km + k_add)
            mask = None
        elif axes == IJ:
            mask = (True, True, False)
            origin = (self.ng, self.ng)
            shape = (2 * self.ng + self.npx, 2 * self.ng + self.npy)
        elif axes == K:
            mask = (False, False, True)
            origin = (0,)
            shape = (self.km + k_add, )
        else:
            raise ValueError("Axes unrecognized")
        return gt4py.storage.empty(
            backend=self._backend, default_origin=origin, shape=shape, dtype=dtype,
            mask=mask)


def do_test(data_file, backend):
    data = Dataset(data_file, backend)

    # Deserialize fields
    field_arg_names = ("hs", "w3", "pt",
                       "delp", "gz", "pef", "ws")
    field_args = {name: data[name] for name in field_arg_names}

    # q_con is stored as a scalar, but is actually a 1D field
    q_con = data.new(K, float, pad_k=True)
    q_con = data["q_con"]

    # pe should be a temporary, but needs to be a field argument for now.
    pe = data.new(IJK, float, pad_k=True)

    # Deserialize scalars
    scalar_arg_names = ("cappa", "p_fac", "scale_m", "ms", "dt", "akap", "cp", "ptop")
    scalar_args = {name: data[name] for name in scalar_arg_names}

    # Deserialize compile-time constants
    compile_time_args_names = {"A_IMP": "a_imp"}
    compile_time_args = {k: data[v] for k, v in compile_time_args_names.items()}

    for k, v in field_args.items():
        assert hasattr(v, "shape"), f"{k} does not have a 'shape' attribute"

    riem = stencil(backend=backend, definition=riem_solver_c,
                   externals=compile_time_args)

    riem(pe=pe, q_con=q_con, **field_args, **scalar_args)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise ValueError("Usage: main.py path/to/dataset.nc [backend]")
    file_name = sys.argv[1]
    if len(sys.argv) > 2:
        backend = sys.argv[2]
    else:
        backend = "numpy"
    do_test(file_name, backend)
