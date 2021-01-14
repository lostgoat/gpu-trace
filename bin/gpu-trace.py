#!/usr/bin/env python3

# MIT License

# Copyright (c) 2019 Valve Corporation

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import sys
import stat
import time
import logging
import argparse
import subprocess
import shutil
import signal
import threading
import _thread
import atexit
from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client


class State(object):
    # Singleton boilerplate
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(State, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    # Members
    traceExitEvent = threading.Event()
    daemon = None
    rpcServerPort = 47317

####################################
# Helpers
####################################
class DeathError(Exception):
    pass


def Die(msg, previousException=None):
    Log.critical(msg)
    if previousException is None or not Log.isEnabledFor(logging.DEBUG):
        sys.exit(-1)
    else:
        raise previousException


# For logging purposes output directly to Log
Log = logging.getLogger('gpu-trace')


def SetupLogging(logPath, logLevel):
    Log.setLevel(logLevel)

    consoleLogFormat = logging.Formatter('%(levelname)s - %(message)s')
    consoleLog = logging.StreamHandler()
    consoleLog.setLevel(logLevel)
    consoleLog.setFormatter(consoleLogFormat)
    Log.addHandler(consoleLog)

    if logPath.strip():
        fileLogFormat = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        fileLog = logging.FileHandler(logPath)
        fileLog.setLevel(logLevel)
        fileLog.setFormatter(fileLogFormat)
        Log.addHandler(fileLog)


def GetBinary(name):
    path = shutil.which(name)

    if not path.strip():
        Die(f"Failed to find binary in PATH: {name}")

    Log.debug(f"Found {name} at {path}")
    return path


def RunCommand(cmd, background=False):
    execCmd = cmd

    # Log.debug( f"Executing {execCmd}" );
    if background:
        subprocess.Popen(cmd)
        return ""
    else:
        cmdProc = subprocess.run(execCmd, capture_output=True)
        # Log.debug( f"return: {cmdProc.returncode}" );
        # Log.debug( f"stdout: {cmdProc.stdout}" );
        # Log.debug( f"stderr: {cmdProc.stderr}" );

        cmdProc.check_returncode()
        return str(cmdProc.stdout, 'utf-8')


def IsFdValid(fd):
    if fd < 0:
        return False

    try:
        os.fstat(fd)
        return True
    except:
        return False

def AddPermissions(path, mask):
    current = os.stat(path).st_mode
    os.chmod(path, current | mask)

####################################
# Managing trace-cmd
####################################
class GpuTrace:
    def __init__(self):
        self.traceCmd = GetBinary('trace-cmd')
        self.captureMask = 0o666
        self.traceCapable = False

        # A bit of sanity checking
        self.EnsureTraceCmdCapable()

        self.TraceSetup()

        self.traceEventArgs = []
        for event in GpuTrace.traceEvents:
            self.traceEventArgs.append("-e")
            self.traceEventArgs.append(f"{event}")

    def __del__(self):
        if self.traceCapable:
            if self.IsTraceEnabled():
                self.StopCapture()

    def TraceSetup(self):
        try:
            AddPermissions("/sys/kernel/tracing/", stat.S_IXOTH)
            AddPermissions("/sys/kernel/tracing/trace_marker", stat.S_IWOTH)
        except Exception as e:
            Die('Failed trace setup, are you root?', e)

    def EnsureTraceCmdCapable(self):
        try:
            self.TraceCmd("stat")
            self.traceCapable = True
        except Exception as e:
            Die("Failed run trace-cmd, are you root?", e)

    def StartCapture(self):
        if self.IsTraceEnabled():
            Log.warning(
                "Attempted to start trace, but one was already running")
            Log.warning("Killing current trace session to start a new one")
            self.StopCapture()
            return

        Log.info("Initializing GPU trace, please wait...")
        self.TraceCmd("start", "-b", "8000", "-D", "-i", self.traceEventArgs)
        Log.info("GPU Trace started")

    def StopCapture(self):
        if not self.IsTraceEnabled():
            Log.error("Attempted to stop trace, but no trace was enabled")
            return

        Log.info("GPU Trace stopping")
        self.TraceCmd("reset")
        self.TraceCmd("snapshot", "-f")
        self.TraceCmd("stop")

    def CaptureTrace(self, path):
        if not self.IsTraceEnabled():
            Log.error("Attempted to capture trace, but no trace was enabled")
            return False

        Log.info(f"GPU Trace capture requested: {path}")
        self.TraceCmd("stop")
        self.TraceCmd("extract", "-k", "-o", path)
        os.chmod(path, self.captureMask)

        Log.debug("GPU Trace capture resuming")
        self.TraceCmd("restart")
        return True

    def TraceCmd(self, *args):
        procArgs = [self.traceCmd]
        for arg in args:
            if isinstance(arg, list):
                procArgs.extend(arg)
            else:
                procArgs.append(arg)
        return RunCommand(procArgs)

    def IsTraceEnabled(self):
        statOutput = self.TraceCmd("stat")
        if "disabled" in statOutput:
            return False
        else:
            return True

    traceEvents = [
        # https://github.com/mikesart/gpuvis/wiki/TechDocs-Linux-Scheduler
        "sched:sched_switch",
        "sched:sched_process_fork",
        "sched:sched_process_exec",
        "sched:sched_process_exit",

        "drm:drm_vblank_event",
        "drm:drm_vblank_event_queued",
        "drm:drm_vblank_event_delivered",

        # https://github.com/mikesart/gpuvis/wiki/TechDocs-AMDGpu
        "amdgpu:amdgpu_vm_flush",
        "amdgpu:amdgpu_cs_ioctl",
        "amdgpu:amdgpu_sched_run_job",
        "amdgpu:amdgpu_ttm_bo_move",
        "*fence:*fence_signaled",

        # https://github.com/mikesart/gpuvis/wiki/TechDocs-Intel
        #
        # NOTE: the i915_gem_request_submit, i915_gem_request_in, i915_gem_request_out
        # tracepoints require the CONFIG_DRM_I915_LOW_LEVEL_TRACEPOINTS Kconfig option to
        # be enabled.
        "i915:i915_flip_request",
        "i915:i915_flip_complete",
        "i915:intel_gpu_freq_change",
        "i915:i915_gem_request_add",
        "i915:i915_gem_request_submit",
        "i915:i915_gem_request_in",
        "i915:i915_gem_request_out",
        "i915:intel_engine_notify",
        "i915:i915_gem_request_wait_begin",
        "i915:i915_gem_request_wait_end",
    ]

####################################
# Signal Handlers
####################################
def SigIntHandler(sig, frame):
    State().traceExitEvent.set()
    if State().daemon is not None:
        State().daemon.Shutdown()


def RegisterSignalHandlers():
    signal.signal(signal.SIGINT, SigIntHandler)

####################################
# Interacting with GpuVis
####################################
class GpuVis:
    def __init__(self):
        self.gpuvis = GetBinary('gpuvis')
        self.user = RunCommand('logname').strip()

    def OpenTrace(self, path):
        RunCommand(["sudo", "-u", self.user, self.gpuvis, path], True)

####################################
# Daemon
####################################
class Daemon:
    def __init__(self, args):
        State().daemon = self

        self.args = args
        self.server = None
        self.gpuTrace = GpuTrace()
        self.capturing = False

        self.RpcServerSetup()
        self.gpuTrace.StartCapture()
        self.capturing = True

    def Run(self):
        Log.info('GPU Trace daemon ready')
        self.server.serve_forever()
        Log.info('GPU Trace daemon exiting')

    def ShutdownWork(self, server):
        if server is not None:
            Log.info("Shutting down rpc server")
            server.shutdown()

    def Shutdown(self):
        Log.info("Daemon shutdown request received")
        _thread.start_new_thread(Daemon.ShutdownWork, (self.server,))
        self.gpuTrace.StopCapture()
        sys.exit()

    def RpcCapture(self, path):
        Log.info(f"Executing capture command: {path}")
        return self.gpuTrace.CaptureTrace(path.strip())

    def RpcStart(self):
        Log.info(f"Executing start command")

        if self.capturing:
            return True

        self.gpuTrace.StartCapture()
        self.capturing = True
        return True

    def RpcStop(self):
        Log.info(f"Executing stop command")

        if not self.capturing:
            return True

        self.gpuTrace.StopCapture()
        self.capturing = False
        return True

    def RpcExit(self):
        Log.info(f"Executing exit command")
        self.Shutdown()
        return True

    def RpcServerSetup(self):
        self.server = SimpleXMLRPCServer(("localhost", State().rpcServerPort))

        self.server.register_function(self.RpcCapture, "capture")
        self.server.register_function(self.RpcStart, "start")
        self.server.register_function(self.RpcStop, "stop")
        self.server.register_function(self.RpcExit, "exit")

####################################
# Daemon client
####################################
def ClientMain(args):
    Log.debug('GPU trace client main')
    rpcServerUrl = f"http://localhost:{State().rpcServerPort}/"

    with xmlrpc.client.ServerProxy(rpcServerUrl) as rpcServer:
        if args.command_capture:
            Log.info(f"Requesting capture to {args.output_dat} ...")
            ret = rpcServer.capture(args.output_dat)

            if not ret:
                Log.info("Capture request failed")
                return False

            if args.open_gpuvis:
                GpuVis().OpenTrace(args.output_dat)
            return True

        if args.command_exit:
            Log.info('Requesting exit...')
            rpcServer.exit()

        if args.command_start:
            Log.info('Requesting start...')
            rpcServer.start()

        if args.command_stop:
            Log.info('Requesting stop...')
            rpcServer.stop()

####################################
# Standalone
####################################
def StandaloneMain(args):
    Log.debug('GPU trace standalone main')

    gpuTrace = GpuTrace()
    gpuTrace.StartCapture()

    State().traceExitEvent.wait()

    gpuTrace.CaptureTrace(args.output_dat)
    GpuVis().OpenTrace(args.output_dat)


####################################
# Main/Input handling
####################################
def Main():
    RegisterSignalHandlers()

    parser = argparse.ArgumentParser(description='GPU profiler capture')

    parser.add_argument('-d', '--daemon', action="store_true",
                        default=False, help="Start in daemon mode")
    parser.add_argument('-v', '--verbose', action="store_true",
                        default=False, help="Enable verbose output")
    parser.add_argument('-l', '--logfile', default="",
                        help="Log all messages to this file")
    parser.add_argument('-o', '--output-dat', dest="output_dat",
                        default="gputrace.dat", help="Trace output filename")
    parser.add_argument('--no-gpuvis', action="store_false", dest="open_gpuvis",
                        default=True, help="Don't open gpuvis when a capture is taken")

    # Rpc commands
    parser.add_argument('--capture', action="store_true", dest="command_capture",
                        default=False, help="Send a capture request to the Daemon. See OUTPUT_DAT for path.")
    parser.add_argument('--exit', action="store_true", dest="command_exit",
                        default=False, help="Send an exit command to the Daemon.")
    parser.add_argument('--start', action="store_true", dest="command_start",
                        default=False, help="Send an start command to the Daemon.")
    parser.add_argument('--stop', action="store_true", dest="command_stop",
                        default=False, help="Send a stop command to the Daemon.")

    args = parser.parse_args()

    # Store the path arguments as full paths
    args.output_dat = os.path.realpath(args.output_dat)

    try:
        logLevel = logging.DEBUG if args.verbose else logging.INFO
        if args.daemon:
            logLevel = logging.DEBUG

        SetupLogging('amdgpu-trace.log', logLevel)
    except Exception as e:
        Die("Failed to setup logging", e)

    Log.debug(f"Daemon mode: {args.daemon}")
    Log.debug(f"Verbose output: {args.verbose}")
    Log.debug(f"Logfile: {args.logfile}")
    Log.debug(f"Output dat: {args.output_dat}")

    if args.daemon:
        daemon = Daemon(args)
        daemon.Run()
    elif args.command_capture or args.command_exit or args.command_start or args.command_stop:
        ClientMain(args)
    else:
        StandaloneMain(args)


if __name__ == "__main__":
    Main()
