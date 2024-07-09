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
import os.path
import sys
import stat
import time
import logging
import argparse
import subprocess
import shutil
import signal
import tempfile
import threading
import _thread
import atexit
from xmlrpc.server import SimpleXMLRPCServer
import xmlrpc.client
import json
import random
import string
import glob
from zipfile import ZipFile


####################################
# Config
####################################
class Config:
    def __init__(self):
        self.jsonData = {}
        self.LoadConfig()

    def GetConfigDir(self):
        fallbackDir = os.path.join(os.getenv('HOME', ''), '.config')
        configRoot = os.path.realpath(os.getenv('XDG_CONFIG_HOME', fallbackDir))
        return os.path.join(configRoot, 'gpu-trace')

    def GetConfigFile(self):
        return os.path.join(self.GetConfigDir(), 'config.json')

    def LoadConfig(self):
        try:
            with open(self.GetConfigFile(), 'r') as f:
                self.jsonData = json.load(f)
        except:
            self.jsonData = {}

    def SaveConfig(self):
        os.makedirs(self.GetConfigDir(), exist_ok=True)
        with open(self.GetConfigFile(), 'w') as f:
            json.dump(self.jsonData, f, indent=4)

    def GetConfigValue(self, key, default):
        if self.jsonData and key in self.jsonData:
            return self.jsonData[key]
        return default

    def SetConfigValue(self, key, val):
        self.jsonData[key] = val
        self.SaveConfig()


####################################
# State singleton
####################################
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
    config = Config()
    rpcServerPort = 47317
    tempFiles = []

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


def TempPath(prefix=None, suffix=None):
    f = tempfile.NamedTemporaryFile(prefix=prefix, suffix=suffix, delete=False)
    p = f.name
    f.close()
    os.chmod(p, 0o666)
    State().tempFiles.append(p)
    return p

def RemoveFile(path):
    if path is None:
        return;

    try:
        os.unlink(path)
    except FileNotFoundError:
        pass

def RenamePath(srcPath, dstPath):
    RemoveFile(dstPath)
    os.rename(srcPath, dstPath);

def CleanupTempFiles():
    for file in State().tempFiles:
        RemoveFile(file)

atexit.register(CleanupTempFiles)

def CreateArchive(out_path, paths):
    with ZipFile(out_path, 'w') as z:
        for p in paths:
            z.write(p, arcname=os.path.basename(p))


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

    if path is None or not path.strip():
        Die(f"Failed to find binary in PATH: {name}")

    Log.debug(f"Found {name} at {path}")
    return path


def RunCommand(cmd, background=False, expectFail=False):
    execCmd = cmd

    # Log.debug( f"Executing {execCmd}" );
    if background:
        return subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True)
    else:
        cmdProc = subprocess.run(execCmd, capture_output=True)
        # Log.debug( f"return: {cmdProc.returncode}" );
        # Log.debug( f"stdout: {cmdProc.stdout}" );
        # Log.debug( f"stderr: {cmdProc.stderr}" );

        if not expectFail:
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
    if not current & mask:
        os.chmod(path, current | mask)

def PerfCmd(*args, background=False):
    procArgs = [PerfCmd.cmd]
    for arg in args:
        if isinstance(arg, list):
            procArgs.extend(arg)
        else:
            procArgs.append(arg)
    return RunCommand(procArgs, background)
PerfCmd.cmd = GetBinary('perf')

def ConvertPerfToJSON(perfCapturePath, jsonPath):
    Log.info(f"Converting trace to JSON")
    try:
        PerfCmd( "data", "convert", "-i", perfCapturePath, "--to-json", jsonPath, "--force")
        return True;
    except Exception as e:
        Log.error(f"Could not convert perf trace to JSON: {e}")
        return False

def CreateGPUVisPackage(ftraceCapturePath, perfCapturePath, outPath):
    capturePaths=[ftraceCapturePath]
    if perfCapturePath is not None:
        jsonPath = TempPath('perf-', '.json')
        if ConvertPerfToJSON(perfCapturePath, jsonPath):
            capturePaths.append(jsonPath)
        else:
            Log.error(f"Failed to convert {perfCapturePath} to json, skipping")
    else:
        Log.info(f"No perf capture detected");

    Log.info(f"Creating package: {outPath}")
    CreateArchive(outPath, capturePaths)

def ProcessCaptureResult(ftraceCapturePath, perfCapturePath, bOpenGpuVis, outPath):
    CreateGPUVisPackage(ftraceCapturePath, perfCapturePath, outPath)
    Log.info(f"Successfully generated trace package: {outPath}")

    if bOpenGpuVis:
        GpuVis().OpenTrace(outPath)

####################################
# Managing trace-cmd
####################################
class GpuTrace:
    def __init__(self):
        self.traceCmd = GetBinary('trace-cmd')
        self.traceCmdVersion = self.GetTraceCmdVersion()
        self.NeedConvert = self.HaveTraceCmdVersion([3, 1, 1])
        self.captureMask = 0o666
        self.traceCapable = False

        # A bit of sanity checking
        self.EnsureTraceCmdCapable()

        self.TraceSetup()

        self.traceEventArgs = ["-C", "perf"]
        for event in GpuTrace.traceEvents:
            self.traceEventArgs.append("-e")
            self.traceEventArgs.append(f"{event}")

    def __del__(self):
        if self.traceCapable:
            if self.IsTraceEnabled():
                self.StopCapture(quiet=True)

    def TraceSetup(self):
        try:
            AddPermissions("/sys/kernel/tracing/", stat.S_IXOTH)
            AddPermissions("/sys/kernel/tracing/", stat.S_IROTH)
            AddPermissions("/sys/kernel/tracing/trace_marker", stat.S_IWOTH)
        except Exception as e:
            Die('Failed trace setup, are you root?', e)

    def GetTraceCmdVersion(self):
        try:
            output = RunCommand(self.traceCmd, False, True)
            for line in output.splitlines():
                if "version" in line:
                    sVersion = line.split(" ")[2].split(".")
                    return [int(component) for component in sVersion]
            return [0,0,0]
        except Exception as e:
            Die("Failed to run trace-cmd, is it installed?", e)

    def HaveTraceCmdVersion(self, queryVersion):
        if self.traceCmdVersion[0] > queryVersion[0]:
            return True

        if self.traceCmdVersion[0] < queryVersion[0]:
            return False

        if self.traceCmdVersion[1] > queryVersion[1]:
            return True

        if self.traceCmdVersion[1] < queryVersion[1]:
            return False

        if self.traceCmdVersion[2] >= queryVersion[2]:
            return True

        return False

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

    def StopCapture(self, *, quiet=False):
        if not self.IsTraceEnabled():
            if not quiet:
                Log.error("Attempted to stop trace, but no trace was enabled")
            return

        if not quiet:
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

        # gpuvis currently only support file version 6
        if self.NeedConvert:
            Log.info(f"GPU Trace needs conversion to compatible format")
            convertSource = path + ".tmp"
            os.rename(path, convertSource)
            self.TraceCmd("convert", "--file-version", "6", "-i", convertSource, "-o", path)

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
# Managing perf
####################################
class PerfTrace:
    class PerfDaemon:
        def __init__(self, proc, filename):
            self.proc = proc
            self.filename = filename

        def __del__(self):
            if self.IsRunning():
                self.Terminate()

        def IsRunning(self):
            return self.proc.poll() is None

        def Terminate(self):
            self.proc.terminate()
            self.proc.wait()

        def RequestCapture(self):
            self.proc.send_signal(signal.SIGUSR2)

        def TraceFile(self):
            r = glob.glob(f"{self.filename}.*")
            return r[0] if len(r) > 0 else None

        def WaitTraceFile(self, delay_ms):
            # Implicitly rounds any nonzero value up to the next
            # greatest multiple of 500ms.

            delay_s = delay_ms * 0.001

            while self.TraceFile() is None and delay_s > 0:
                time.sleep(0.5)
                delay_s -= 0.5

            return self.TraceFile()

    def __init__(self):
        self.perfCmd = GetBinary('perf')
        self.captureMask = 0o666
        self.perfCapable = False
        self.perfDaemon = None

        # Ensure that an appropriate version of perf is available
        # and that we can run a perf trace.
        self.perfCapable = self.EnsurePerfCapable()

    def __del__(self):
        if self.perfCapable:
            if self.IsRecordEnabled():
                self.StopRecord(quiet=True)

    def EnsurePerfCapable(self):
        try:
            res = subprocess.run(
                [self.perfCmd, "data", "convert", "--help"],
                capture_output=True)
            if res.stderr.find(b"--to-json") < 0:
                Log.error("No perf data convert --to-json support")
                return False
        except Exception as e:
            Log.error("Failed run perf. Is it installed?")
            return False

        try:
            # This will take some time, but it shouldn't be longer than
            # a second or two.
            PerfCmd("record", "--sample-cpu", "-Fmax", "-o/dev/null", "--", "echo")
        except Exception as e:
            Log.error("Failed run perf record, are you root?")
            return False

        return True

    def StartRecord(self, restart=False):
        if self.IsRecordEnabled():
            if not restart:
                Log.warning(
                    "Attempted to start perf, but it is already running")
                Log.warning("Killing current perf session to start a new one")
            self.StopRecord()

        Log.info("Initializing perf record, please wait...")

        # Get a new temporary prefix. We specifically want the
        # deprecated mktemp style semantics here, since we don't
        # want/need an open file pointer and since perf record
        # will append a timestamp anyway, so a custom method
        # to generate a random temporary file prefix is used.
        filename = self.NewTempPrefix()

        self.perfDaemon = self.PerfDaemon(
            PerfCmd(
                "record", "--sample-cpu", "-Fmax", "-m16M", "--overwrite",
                "--call-graph", "fp", "--switch-output", "--switch-max-files",
                "1", "-o", filename, background=True),
            filename
        )

        if self.IsRecordEnabled():
            Log.info("Perf record started")
        else:
            Die("Failed to successfully start perf recording. Aborting")

    def StopRecord(self, *, quiet=False):
        if not self.IsRecordEnabled():
            if not quiet:
                Log.error("Attempted to stop recording, but recording not enabled")
            return

        if not quiet:
            Log.info("perf record stopping...")

        self.perfDaemon.Terminate()

        # perf always spits out one last file and there's currently
        # no way to disable this. We don't want this file or care
        # about it, so try to simply remove it.
        filename = self.perfDaemon.TraceFile()
        try:
            os.remove(filename)
        except (FileNotFoundError, TypeError) as e:
            if not quiet:
                Log.warning(
                    "Could not delete final data file as it does not exist")
                Log.warning(f"May be leaking {filename}")

        # For some reason, perf also creates an empty file
        # with the prefix name we've provided. This file is not
        # needed for anything, so if happens to exist remove it here.
        if os.path.isfile(self.perfDaemon.filename):
            os.remove(self.perfDaemon.filename)

        self.perfDaemon = None

        if not quiet:
            Log.info("perf record stopped")

    def CaptureTrace(self, path):
        if not self.IsRecordEnabled():
            Log.error("Attempted perf trace capture, but no trace was enabled")
            return False

        Log.info(f"perf trace capture requested: {path}")
        Log.info("Requesting capture. This may take some time")

        self.perfDaemon.RequestCapture()
        filename = self.perfDaemon.WaitTraceFile(15000)

        if not isinstance(filename, str):
            Log.error("Failed to capture trace.")
            # Unfortunately, it is necessary to switch to another
            # temporary filename here to avoid a race condition
            # on future captures.
            Log.warn("Force restarting as a precaution.")
            self.StartRecord(restart=True)
            return False

        # Rename our output file to the desired output file name
        try:
            RenamePath(filename, path)
            os.chmod(path, self.captureMask)
        except Exception as e:
            Log.error(f"Failed to rename perf capture: {e}")
            return False;

        Log.info("perf trace file written to requested path")
        return True

    def IsRecordEnabled(self):
        return self.perfDaemon is not None and self.perfDaemon.IsRunning()

    def NewTempPrefix(self):
        r = ''.join(random.choice(string.ascii_letters) for _ in range(10))
        return f"/tmp/perf.{r}.dat"

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
        gpuvis_args = ["sudo", "-u", self.user, self.gpuvis, path]
        RunCommand(gpuvis_args, True)

####################################
# Daemon
####################################
class Daemon:
    CAPTURE_SUCCESS = 0
    CAPTURE_FAILURE = 1

    def __init__(self, args):
        State().daemon = self

        self.args = args
        self.server = None
        self.gpuTrace = GpuTrace()
        self.perfTrace = PerfTrace()
        self.capturing = False
        self.captureMask = 0o666

        if not self.perfTrace.perfCapable:
            Log.warning(
                "Failed to verify perf capability. Disabling perf recording.")

        self.RpcServerSetup()

        if State.config.GetConfigValue('StartupCapture', False):
            self.RpcStart(quiet=True)
        else:
            Log.info('Startup tracing is disabled')

    def Run(self):
        Log.info('GPU Trace daemon ready')
        self.server.serve_forever()
        Log.info('GPU Trace daemon exiting')

    @staticmethod
    def ShutdownWork(server):
        if server is not None:
            Log.info("Shutting down rpc server")
            server.shutdown()

    def Shutdown(self):
        Log.info("Daemon shutdown request received")
        _thread.start_new_thread(Daemon.ShutdownWork, (self.server,))
        self.RpcStop(quiet=True)
        sys.exit()

    def CleanupIntermediates(self):
        CleanupTempFiles()

    def RpcCapture(self):
        Log.info(f"Executing capture command")

        self.CleanupIntermediates();

        dictRet = {}
        tracePath = TempPath('gputrace-', '.dat')
        ok = self.gpuTrace.CaptureTrace(tracePath)
        dictRet[ "ftracepath" ] = tracePath;

        if ok and self.perfTrace.perfCapable:
            perfPath = TempPath('perf-', '.perf')
            ok = self.perfTrace.CaptureTrace(perfPath)
            dictRet[ "perftracepath" ] = perfPath;
        else:
            Log.info(f"Skipping perf trace capture: not capable")

        dictRet[ "retcode" ] = Daemon.CAPTURE_SUCCESS if ok else Daemon.CAPTURE_FAILURE;
        return dictRet;

    def RpcStart(self, *, quiet=False):
        if not quiet:
            Log.info(f"Executing start command")

        if self.capturing:
            return True

        self.gpuTrace.StartCapture()
        if self.perfTrace.perfCapable:
            self.perfTrace.StartRecord()
        self.capturing = True

        return True

    def RpcStop(self, *, quiet=False):
        if not quiet:
            Log.info(f"Executing stop command")

        if not self.capturing:
            return True

        self.gpuTrace.StopCapture()
        if self.perfTrace.perfCapable:
            self.perfTrace.StopRecord()
        self.capturing = False

        return True

    def RpcExit(self):
        Log.info(f"Executing exit command")
        self.Shutdown()
        return True

    def RpcGetTracingStatus(self):
        Log.info(f"Executing get tracing status command")
        return self.capturing

    def RpcCleanup(self):
        Log.info(f"Executing cleanup command")
        self.CleanupIntermediates()
        return True

    def RpcServerSetup(self):
        self.server = SimpleXMLRPCServer(("localhost", State().rpcServerPort))

        self.server.register_function(self.RpcCapture, "capture")
        self.server.register_function(self.RpcStart, "start")
        self.server.register_function(self.RpcStop, "stop")
        self.server.register_function(self.RpcExit, "exit")
        self.server.register_function(self.RpcGetTracingStatus, "getTracingStatus")
        self.server.register_function(self.RpcCleanup, "cleanup")

####################################
# Daemon client
####################################
def ClientMain(args):
    Log.debug('GPU trace client main')
    rpcServerUrl = f"http://localhost:{State().rpcServerPort}/"

    with xmlrpc.client.ServerProxy(rpcServerUrl) as rpcServer:
        if args.command_capture:
            outPath = args.output.strip();
            Log.info(f"Requesting capture: output: {outPath}")

            ret = rpcServer.capture()
            retcode = ret[ "retcode" ];
            ftraceCapturePath = ret.get( "ftracepath" );
            perfCapturePath = ret.get( "perftracepath" );

            if retcode is not Daemon.CAPTURE_SUCCESS:
                Log.error(f"Failed to capture trace: response: {ret}")

            ProcessCaptureResult(ftraceCapturePath, perfCapturePath, args.open_gpuvis, outPath)
            rpcServer.cleanup()
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

        if args.command_get_tracing_status:
            print( "1" if rpcServer.getTracingStatus() else "0" )
            

####################################
# Standalone
####################################
def StandaloneMain(args):
    Log.debug('GPU trace standalone main')

    gpuTrace = GpuTrace()
    gpuTrace.StartCapture()

    perfTrace = PerfTrace()

    if perfTrace.perfCapable:
        perfTrace.StartRecord()
    else:
        Log.warning(
            "Failed to verify perf capability. Disabling perf recording.")

    State().traceExitEvent.wait()

    tracePath = TempPath('gputrace-', '.dat')
    ok = gpuTrace.CaptureTrace(tracePath)

    perfPath = None
    if ok and perfTrace.perfCapable:
        perfPath = TempPath('perf-', '.json')
        ok = perfTrace.CaptureTrace(perfPath)

    if not ok:
        Log.error( "Failed to capture trace" )

    ProcessCaptureResult(tracePath, perfPath, args.open_gpuvis, args.output.strip() )
    RemoveFile(tracePath)
    RemoveFile(perfPath)

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
    parser.add_argument('-o', '--output', dest="output",
                        default="gpu-trace.zip", help="Trace output filename")
    parser.add_argument('--no-gpuvis', action="store_false", dest="open_gpuvis",
                        default=True, help="Don't open gpuvis when a capture is taken")

    # Config commands
    parser.add_argument('--enable-startup-tracing', action=argparse.BooleanOptionalAction,
                        help="Enable/disable tracing on service startup")
    parser.add_argument('--get-startup-tracing', action="store_true",
                        help="Check if the service will start tracing on startup")

    # Rpc commands
    parser.add_argument('--capture', action="store_true", dest="command_capture",
                        default=False, help="Send a capture request to the Daemon. See OUTPUT_DAT and PERF_JSON for paths.")
    parser.add_argument('--exit', action="store_true", dest="command_exit",
                        default=False, help="Send an exit command to the Daemon.")
    parser.add_argument('--start', action="store_true", dest="command_start",
                        default=False, help="Send an start command to the Daemon.")
    parser.add_argument('--stop', action="store_true", dest="command_stop",
                        default=False, help="Send a stop command to the Daemon.")
    parser.add_argument('--get-tracing-status', action="store_true", dest="command_get_tracing_status",
                        default=False, help="Query if the daemon is currently tracing or not")

    args = parser.parse_args()

    # Store the path arguments as full paths
    args.output = os.path.realpath(args.output)

    if not args.logfile:
        if args.daemon:
            args.logfile = '/var/log/gpu-trace-daemon.log'
        else:
            args.logfile = 'gpu-trace.log'

    try:
        logLevel = logging.DEBUG if args.verbose else logging.INFO
        if args.daemon:
            logLevel = logging.DEBUG

        SetupLogging(args.logfile, logLevel)
    except Exception as e:
        Die("Failed to setup logging", e)

    Log.debug(f"Daemon mode: {args.daemon}")
    Log.debug(f"Verbose output: {args.verbose}")
    Log.debug(f"Logfile: {args.logfile}")
    Log.debug(f"Output path: {args.output}")
    Log.debug(f"Config file: {State().config.GetConfigFile()}")

    if args.daemon:
        daemon = Daemon(args)
        daemon.Run()
    elif args.command_capture or args.command_exit or args.command_start or args.command_stop or args.command_get_tracing_status:
        ClientMain(args)
    elif args.enable_startup_tracing is not None:
        State().config.SetConfigValue("StartupCapture", args.enable_startup_tracing)
    elif args.get_startup_tracing:
        print( "1" if State().config.GetConfigValue("StartupCapture", None) else "0" )
    else:
        StandaloneMain(args)


if __name__ == "__main__":
    Main()
