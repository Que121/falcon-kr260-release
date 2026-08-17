#!/usr/bin/env python
"""KR260 serial-console rescue: log in over UART and fix the board's network config.

Expect-style dialogue over COM port (115200 8N1): wake the console, log in if needed,
stop the DHCP-looping network managers, put a static IP on the cabled NIC (eth1), and
print the resulting ip/route state. Idempotent — safe to re-run.

Usage: python serial_rescue.py [--port COM3] [--user ubuntu] [--password PW]
       [--iface eth1] [--ip 192.168.137.218/24] [--probe-only]
"""
import argparse, sys, time
import serial

def read_for(ser, seconds):
    buf = b""
    end = time.time() + seconds
    while time.time() < end:
        chunk = ser.read(4096)
        if chunk:
            buf += chunk
        else:
            time.sleep(0.05)
    return buf.decode(errors="replace")

def send(ser, line, wait=1.0):
    ser.write((line + "\n").encode())
    ser.flush()
    return read_for(ser, wait)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--user", default="ubuntu")
    ap.add_argument("--password", required=True)
    ap.add_argument("--iface", default="eth1")
    ap.add_argument("--ip", default="192.168.137.218/24")
    ap.add_argument("--probe-only", action="store_true", help="just wake the console and show state")
    args = ap.parse_args()

    ser = serial.Serial(args.port, 115200, timeout=0.2)
    log = []

    def step(desc, out):
        log.append(f"--- {desc} ---\n{out}")
        print(f"--- {desc} ---\n{out}", flush=True)

    # Wake the console (a couple of newlines) and look at what we get
    ser.write(b"\n"); ser.flush()
    out = read_for(ser, 2.0)
    ser.write(b"\n"); ser.flush()
    out += read_for(ser, 2.0)
    step("wake", out)

    # Login dance (handle: still-booting, already-logged-in shell, login: prompt, Password: prompt)
    deadline = time.time() + 240  # board may still be booting; be patient
    reached = False
    while time.time() < deadline:
        tail = out[-400:]
        if "login:" in tail:
            out = send(ser, args.user, wait=2.5); step("sent user", out)
        elif "Password:" in tail or "password:" in tail:
            out = send(ser, args.password, wait=4.0); step("sent password", out)
        elif tail.rstrip().endswith("$") or tail.rstrip().endswith("#") or "~$" in tail or ":~" in tail:
            reached = True; break
        else:
            ser.write(b"\n"); ser.flush(); out = read_for(ser, 4.0)
            if out.strip():
                step("nudge", out[-300:])
    if not reached:
        print("FATAL: never reached a shell prompt (board may still be booting — re-run me)", flush=True)
        sys.exit(2)

    out = send(ser, "whoami; hostname", wait=2.0); step("whoami", out)
    out = send(ser, "ip -br link; ip -br addr; ip route", wait=3.0); step("state before", out)

    if not args.probe_only:
        # sudo may prompt for password
        out = send(ser, "sudo systemctl stop NetworkManager systemd-networkd 2>/dev/null; echo NM_STOPPED", wait=3.0)
        if "password" in out.lower():
            out += send(ser, args.password, wait=4.0)
        step("stop managers", out)
        for cmd in (
            "sudo ip addr flush dev eth0",
            f"sudo ip addr flush dev {args.iface}",
            f"sudo ip addr add {args.ip} dev {args.iface}",
            f"sudo ip link set {args.iface} up",
        ):
            out = send(ser, cmd, wait=2.0)
            if "password" in out.lower():
                out += send(ser, args.password, wait=3.0)
            step(cmd, out)
        out = send(ser, "ip -br addr; ip route; echo RESCUE_DONE", wait=3.0)
        step("state after", out)

    ser.close()

if __name__ == "__main__":
    main()
