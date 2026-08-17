#!/usr/bin/env python3
"""KR260 helper (paramiko from Windows). ubuntu@192.168.137.218.
   python _board.py run "<cmd>" [timeout]      - run a shell command
   python _board.py srun "<cmd>" [timeout]     - run via sudo (pw piped)
   python _board.py put <local> <remote>
   python _board.py get <remote> <local>
   DPU recipe is prepended for run/srun: export XILINX_XRT=/usr + pynq-venv.
"""
import sys, paramiko
HOST, USER, PW = "192.168.137.218", "ubuntu", "ivip4107"
VENV = "export XILINX_XRT=/usr && source /usr/local/share/pynq-venv/bin/activate && "

def cli():
    c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PW, timeout=30)
    return c

def main():
    what = sys.argv[1]; c = cli()
    if what == "put":
        sf = c.open_sftp(); sf.put(sys.argv[2], sys.argv[3]); sf.close()
        print("PUT -> board:%s OK" % sys.argv[3]); c.close(); return
    if what == "get":
        sf = c.open_sftp(); sf.get(sys.argv[2], sys.argv[3]); sf.close()
        print("GET board:%s OK" % sys.argv[2]); c.close(); return
    cmd = sys.argv[2]; tmo = int(sys.argv[3]) if len(sys.argv) > 3 else 120
    if what == "srun":
        cmd = "sudo -S bash -c " + repr(VENV + cmd)
        si, so, se = c.exec_command(cmd, timeout=tmo, get_pty=True)
        si.write(PW + "\n"); si.flush()
    else:
        si, so, se = c.exec_command("bash -c " + repr(VENV + cmd), timeout=tmo)
    out = so.read().decode("utf-8", "replace"); err = se.read().decode("utf-8", "replace")
    sys.stdout.write(out)
    if err.strip(): sys.stderr.write("\n--STDERR--\n" + err)
    c.close()

if __name__ == "__main__":
    main()
