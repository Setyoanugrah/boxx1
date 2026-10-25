#!/usr/bin/env python3
"""RakNet unconnected ping — cek server Bedrock hidup dan baca MOTD-nya.

  python3 mcping.py <host> [port]

Exit 0 kalau dapat PONG, 1 kalau timeout.
"""
import socket, struct, sys, time, os

host = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1"
port = int(sys.argv[2]) if len(sys.argv) > 2 else 19132

magic = bytes.fromhex("00ffff00fefefefefdfdfdfd12345678")
ts = int(time.time() * 1000) & 0xFFFFFFFFFFFFFFFF
pkt = b"\x01" + struct.pack(">Q", ts) + magic + os.urandom(8)

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.settimeout(8)
s.sendto(pkt, (host, port))
try:
    data, addr = s.recvfrom(4096)
except socket.timeout:
    print("TIMEOUT %s:%d" % (host, port))
    sys.exit(1)

if data[0] != 0x1C:
    print("opcode tak terduga 0x%02x" % data[0])
    sys.exit(1)

ln = struct.unpack(">H", data[33:35])[0]
pong = data[35:35 + ln].decode("utf-8", "replace")
print(pong)
