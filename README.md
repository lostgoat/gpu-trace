# gpu-trace

A tool intended to aid the collection process of gpuvis traces.

It can run either as a daemon or standalone fashion.

## Installation instructions

```
cd {path where project was clone}
sudo make install
```

## Standalone capture

```
sudo gpu-trace
# Press Ctrl+C to stop capture and open report in gpuvis
```

## Daemon based capture

First enable the daemon after installation:
```
sudo systemctl enable gpu-trace
sudo systemctl start gpu-trace
```

Capturing a trace:
```
gpu-trace --capture
# Or gpu-trace --capture -o ./capture.dat
```

In daemon mode a trace will always be collected in the background. To temporarily disable capture:
```
gpu-trace --stop
# Do things
gpu-trace --start
```

Alternatively, the daemon can stopped/started using systemctl.
