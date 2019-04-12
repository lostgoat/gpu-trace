# amdgpu-trace

Helper script to simplify trace collection for amdgpu and integration with
reporting tools.

We recommend using gpuvis for visualizing amdgpu trace files. Clone, build and
install this repository:
https://github.com/mikesart/gpuvis

Usage:
```
amdgpu-trace --vis
(press Ctrl+C to stop trace)
```

Text report (no gpuvis installed):
```
amdgpu-trace -r
```

For more options see:
```
amdgpu-trace --help
```
