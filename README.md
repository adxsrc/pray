# `PRay`

`PRay` is a GPU-accelerated geodesic integrator for performing general
relativistic ray tracing for accreting black holes.
`PRay` is written in python and uses
[Google's JAX](https://github.com/google/jax)
package for performance and autodiff.
While not as fast as C codes such as
[`GRay`](https://github.com/luxsrc/gray),
`PRay` is easily extensible and provides a lot of flexibility.
