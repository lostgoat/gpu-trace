#!/usr/bin/env python3

import os
import time
import argparse
import logging

Log = logging.getLogger( 'gpu-trace' );

g_bDaemonShouldExit = False;

class DeathError( Exception ):
	pass

def Die( msg ):
	Log.critical( msg )
	raise DeathError

def Die( e, msg ):
	Log.critical( msg )
	raise e

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

def DaemonMain( args ):
	Log.info( 'GPU trace daemon starting' );

	while not g_bDaemonShouldExit:
		Log.info( 'Looping' )
		time.sleep(60)

def ClientMain( args ):
	Log.debug( 'GPU trace client main' );


def StandaloneMain( args ):
	Log.debug( 'GPU trace standalone main' );


def Main():
	parser = argparse.ArgumentParser( description='GPU profiler capture' )

	parser.add_argument( '-d', '--daemon', action="store_true", default=False, help="Start in daemon mode" )
	parser.add_argument( '-v', '--verbose', action="store_true", default=False, help="Enable verbose output" )
	parser.add_argument( '-l', '--logfile', default="", help="Log all messages to this file" )
	parser.add_argument( '-p', '--wait-for-pid', dest="wait_pid", type=int, default=-1, help="Trace for the lifetime of the specified pid" )
	parser.add_argument( '-o', '--output-dat', dest="output_dat", default="gputrace.dat", help="Trace output filename" )
	parser.add_argument( '-c', '--command', default="", help="Send COMMAND to daemon" )

	args = parser.parse_args()

	try:
		logLevel = logging.DEBUG if args.verbose else logging.INFO
		if args.daemon:
			logLevel = logging.DEBUG;

		SetupLogging( 'amdgpu-trace.log', logLevel );
	except Exception as e:
		Die( e, "Failed to setup logging" );

	Log.debug( f"Daemon mode: {args.daemon}" );
	Log.debug( f"Verbose output: {args.verbose}" );
	Log.debug( f"Logfile: {args.logfile}" );
	Log.debug( f"Tracing lifetime pid: {args.wait_pid}" );
	Log.debug( f"Output dat: {args.output_dat}" );

	if args.daemon:
		DaemonMain( args )
	elif args.command.strip():
		ClientMain( args )
	else:
		StandaloneMain( args )

if __name__ == "__main__":
	Main()
