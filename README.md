# amdgpu-trace

Helper script to simplify trace collection for amdgpu and integration with
reporting tools.

We recommend using gpuvis for visualizing amdgpu trace files. Clone, build and
install this repository:
https://github.com/mikesart/gpuvis

Recommended usage:
```
amdgpu-trace -d trace.dat && gpuvis trace.dat
```

Lazy usage:
```
amdgpu-trace --vis
```

Text report (no gpuvis installed):
```
amdgpu-trace -r
```

For more options see:
```
amdgpu-trace --help
```
