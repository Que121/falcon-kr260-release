#!/usr/bin/env python3
"""Helper: run a command on HPC (via Pro6000 jump) or on Pro6000 directly.
   Usage: python _jump.py HPC "<cmd>"   |   python _jump.py pro "<cmd>"
   Jump path: Windows --paramiko--> Pro6000(REDACTED) --ssh HPC--> HPC.
"""
import sys, paramiko, time

PRO = ("REDACTED", "ANON", "020128")

def pro_client():
    c = paramiko.SSHClient(); c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(PRO[0], username=PRO[1], password=PRO[2], timeout=30)
    return c

def run(c, cmd, tmo=120):
    si, so, se = c.exec_command(cmd, timeout=tmo)
    out = so.read().decode("utf-8", "replace"); err = se.read().decode("utf-8", "replace")
    return out, err

def main():
    where = sys.argv[1]
    c = pro_client()
    if where == "put":                          # python _jump.py put <local> <remote-on-pro>
        sf = c.open_sftp(); sf.put(sys.argv[2], sys.argv[3]); sf.close()
        print("PUT %s -> pro:%s OK" % (sys.argv[2], sys.argv[3])); c.close(); return
    if where == "get":                          # python _jump.py get <remote-on-pro> <local>
        sf = c.open_sftp(); sf.get(sys.argv[2], sys.argv[3]); sf.close()
        print("GET pro:%s -> %s OK" % (sys.argv[2], sys.argv[3])); c.close(); return
    cmd = sys.argv[2]
    if where == "HPC":
        # wrap so the remote ssh gets no stdin
        full = "ssh HPC " + repr(cmd) + " < /dev/null"
        out, err = run(c, full, tmo=int(sys.argv[3]) if len(sys.argv) > 3 else 120)
    else:
        out, err = run(c, cmd, tmo=int(sys.argv[3]) if len(sys.argv) > 3 else 120)
    sys.stdout.write(out)
    if err.strip(): sys.stderr.write("\n--STDERR--\n" + err)
    c.close()

if __name__ == "__main__":
    main()
