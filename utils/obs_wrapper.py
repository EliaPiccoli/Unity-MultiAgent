#! /usr/bin/python
import getpass
import os
import sys
import subprocess
import tempfile
import time

# fd of the file where we print obs things
fd = 0

def main():
    if len(sys.argv) <= 1:
        print("USAGE: obs [--startrecording|--stoprecording]")
        sys.exit(1)
    elif sys.argv[1] == "--startrecording":
        start_recording()
    else:
        stop_recording()


def start_recording():
    fd = os.open("obs_output", os.O_WRONLY | os.O_CREAT)

    # For unixish systems:
    args = ["obs"]
    cwd = None

    # For Windows:
    if sys.platform == "win32":
        cwd = r"C:\Program Files (x86)\obs-studio\bin\64bit"
        args = [os.path.join(cwd, "obs64.exe")]

    args.append("--startrecording")
    args.append("--minimize-to-tray")

    p = subprocess.Popen(
            args=args,
            cwd=cwd,
            stdin = fd,
            stdout = fd,
            stderr = fd
        )

    fn = os.path.join(tempfile.gettempdir(), "obs.{}.pid" .format(getpass.getuser()))
    f = open(fn, "w")
    f.write(str(p.pid))
    f.close()


def stop_recording():
    fn = os.path.join(tempfile.gettempdir(), "obs.{}.pid".format(getpass.getuser()))
    if not os.path.exists(fn):
        print("OBS PID file does not exists: {}".format(fn))
        sys.exit(2)
    f = open(fn, "r")
    pid = f.read()
    f.close()

    # For unixish systems:
    args = ["kill", pid]
    # For Windows:
    if sys.platform == "win32":
        args = ["taskkill", "/f", "/pid", pid]
    subprocess.Popen(args=args).communicate()
    
    os.close(fd)
    os.remove("obs_output")
    os.remove(fn)


if __name__ == "__main__":
    main()
