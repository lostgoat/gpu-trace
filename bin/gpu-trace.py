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
import time
import logging
import argparse
import subprocess
import shutil
import signal
import threading

class State(object):
	# Singleton boilerplate
	_instance = None;
	def __new__( cls, *args, **kwargs ):
		if not cls._instance:
			cls._instance = super( State, cls ).__new__( cls, *args, **kwargs );
		return cls._instance;

	#Members
	traceExitEvent = threading.Event();

####################################
# Helpers
####################################
class DeathError( Exception ):
	pass

def Die( msg, previousException=None ):
	Log.critical( msg )
	if previousException is None or not Log.isEnabledFor( logging.DEBUG ):
		sys.exit( -1 );
	else:
		raise previousException

# For logging purposes output directly to Log
Log = logging.getLogger( 'gpu-trace' );

def SetupLogging( logPath, logLevel ):
	Log.setLevel( logLevel );

	consoleLogFormat = logging.Formatter( '%(levelname)s - %(message)s' );
	consoleLog = logging.StreamHandler();
	consoleLog.setLevel( logLevel );
	consoleLog.setFormatter( consoleLogFormat );
	Log.addHandler( consoleLog );

	if logPath.strip():
		fileLogFormat = logging.Formatter( '%(asctime)s - %(levelname)s - %(message)s' );
		fileLog = logging.FileHandler( logPath );
		fileLog.setLevel( logLevel );
		fileLog.setFormatter( fileLogFormat );
		Log.addHandler( fileLog );

def GetBinary( name ):
	path = shutil.which( name );

	if not path.strip():
		Die( f"Failed to find binary in PATH: {name}" );

	Log.debug( f"Found {name} at {path}" );
	return path

def RunCommand( cmd ):
	execCmd = cmd;
	cmdProc = subprocess.run( execCmd, capture_output=True );
	Log.debug( f"Executed {execCmd}" );
	Log.debug( f"return: {cmdProc.returncode}" );
	Log.debug( f"stdout: {cmdProc.stdout}" );
	Log.debug( f"stderr: {cmdProc.stderr}" );

	cmdProc.check_returncode();
	return str( cmdProc.stdout, 'utf-8' );

####################################
# Managing trace-cmd
####################################
class GpuTrace:
	def __init__( self ):
		self.traceCmd = GetBinary( 'trace-cmd' );

		# A bit of sanity checking
		self.EnsureTraceCmdCapable();

		self.traceEventArgs = [];
		for event in GpuTrace.traceEvents:
			self.traceEventArgs.append( "-e" );
			self.traceEventArgs.append( f"{event}" );

	def __del__( self ):
		self.StopCapture();

	def EnsureTraceCmdCapable( self ):
		try:
			self.TraceCmd( "stat" );
		except Exception as e:
			Die( "Failed run trace-cmd, are you root?", e );

	def StartCapture( self ):
		if self.IsTraceEnabled():
			Log.warning( "Attempted to start trace, but one was already running" );
			Log.warning( "Killing current trace session to start a new one" );
			self.StopCapture();
			return;

		Log.info( "Initializing GPU trace, please wait..." );
		self.TraceCmd( "start", "-b", "8000", "-D", "-i", self.traceEventArgs );
		Log.info( "GPU Trace started" );

	def StopCapture( self ):
		if not self.IsTraceEnabled():
			Log.error( "Attempted to stop trace, but no trace was enabled" );
			return;

		Log.info( "GPU Trace stopping" );
		self.TraceCmd( "reset" );
		self.TraceCmd( "snapshot", "-f" );
		self.TraceCmd( "stop" );

	def CaptureTrace( self, path ):
		if not self.IsTraceEnabled():
			Log.error( "Attempted to capture trace, but no trace was enabled" );
			return;

		Log.info( f"GPU Trace capture requested: {path}" );
		self.TraceCmd( "stop" );
		self.TraceCmd( "extract", "-k", "-o", path );

		Log.debug( "GPU Trace capture resuming" );
		self.TraceCmd( "restart" );

	def TraceCmd( self, *args ):
		procArgs = [ self.traceCmd ];
		for arg in args:
			if isinstance( arg, list ):
				procArgs.extend( arg );
			else:
				procArgs.append( arg );
		return RunCommand( procArgs );

	def IsTraceEnabled( self ):
		statOutput = self.TraceCmd( "stat" );
		if "disabled" in statOutput:
			return False;
		else:
			return True;

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
		];

####################################
# Signal Handlers
####################################
def SigIntHandler( sig, frame ):
	State().traceExitEvent.set();

def RegisterSignalHandlers():
	signal.signal( signal.SIGINT, SigIntHandler )

####################################
# Interacting with GpuVis
####################################
class GpuVis:
	def __init__( self ):
		self.gpuvis = GetBinary( 'gpuvis' );

	def OpenTrace( self, path ):
		return RunCommand( [ self.gpuvis, path ]  );

####################################
# Daemon
####################################
def DaemonMain( args ):
	Log.info( 'GPU trace daemon starting' );

	gpuTrace = GpuTrace();
	gpuTrace.StartCapture();

	State().traceExitEvent.wait();

####################################
# Daemon client
####################################
def ClientMain( args ):
	Log.debug( 'GPU trace client main' );

####################################
# Standalone
####################################
def StandaloneMain( args ):
	Log.debug( 'GPU trace standalone main' );

	gpuTrace = GpuTrace();
	gpuTrace.StartCapture();

	State().traceExitEvent.wait();

	gpuTrace.CaptureTrace( args.output_dat );
	GpuVis().OpenTrace( args.output_dat );


####################################
# Main/Input handling
####################################
def Main():
	RegisterSignalHandlers();

	parser = argparse.ArgumentParser( description='GPU profiler capture' )

	parser.add_argument( '-d', '--daemon', action="store_true", default=False, help="Start in daemon mode" )
	parser.add_argument( '-v', '--verbose', action="store_true", default=False, help="Enable verbose output" )
	parser.add_argument( '-l', '--logfile', default="", help="Log all messages to this file" )
	parser.add_argument( '-o', '--output-dat', dest="output_dat", default="gputrace.dat", help="Trace output filename" )
	parser.add_argument( '-c', '--command', default="", help="Send COMMAND to daemon" )

	args = parser.parse_args()

	try:
		logLevel = logging.DEBUG if args.verbose else logging.INFO
		if args.daemon:
			logLevel = logging.DEBUG;

		SetupLogging( 'amdgpu-trace.log', logLevel );
	except Exception as e:
		Die( "Failed to setup logging", e );

	Log.info( f"Event: {State().traceExitEvent.is_set()}" );
	Log.debug( f"Daemon mode: {args.daemon}" );
	Log.debug( f"Verbose output: {args.verbose}" );
	Log.debug( f"Logfile: {args.logfile}" );
	Log.debug( f"Output dat: {args.output_dat}" );
	Log.debug( f"Client command: {args.command}" );

	if args.daemon:
		DaemonMain( args )
	elif args.command.strip():
		ClientMain( args )
	else:
		StandaloneMain( args )

if __name__ == "__main__":
	Main()
