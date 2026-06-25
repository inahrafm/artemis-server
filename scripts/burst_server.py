#!/usr/bin/env python3
"""
burst_server.py — Artemis Field Measurement (jalankan di HOMESERVER)
Menerima payload dari Raspi, langsung kirim ACK.

Usage:
    python3 burst_server.py --port 9999

Jalankan SEKALI di homeserver sebelum berangkat ke lapangan.
Biarkan terus jalan selama pengukuran.
"""

import argparse
import socket
import threading
from datetime import datetime


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9999)
    p.add_argument("--host", default="0.0.0.0")
    return p.parse_args()


def handle_client(conn, addr):
    try:
        # Terima header: ukuran payload (4 bytes)
        raw_size = b""
        while len(raw_size) < 4:
            chunk = conn.recv(4 - len(raw_size))
            if not chunk:
                return
            raw_size += chunk

        payload_size = int.from_bytes(raw_size, byteorder='big')

        # Terima payload
        received = 0
        while received < payload_size:
            chunk = conn.recv(min(65536, payload_size - received))
            if not chunk:
                break
            received += len(chunk)

        # Kirim ACK
        conn.sendall(b'\x01')

        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] {addr[0]}:{addr[1]} — {payload_size/1024:.1f} KB received, ACK sent")

    except Exception as e:
        print(f"  [ERR] {addr}: {e}")
    finally:
        conn.close()


def main():
    args = parse_args()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(10)

        print(f"\n{'='*50}")
        print(f"  Artemis Burst Server — listening on :{args.port}")
        print(f"  Ctrl+C untuk stop")
        print(f"{'='*50}\n")

        try:
            while True:
                conn, addr = srv.accept()
                t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
                t.start()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
