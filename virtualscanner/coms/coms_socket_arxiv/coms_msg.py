"""
This script defines tx and rx operations
Parameters
----------
    thread object

Performs
--------
   tx to client
   rx from client

Returns
-------
    payload or status

Unit Test app
-------------
     utest_coms
Author: Sairam Geethanath
Date: 03/11/2019
Version 0.0
"""
import json
import struct


def send(self):  # should conform to the class of class ClientThread(Thread):
    # Prefix each message with a 4-byte length (network byte order)
    # Need to change the action key to rx for the receive end to work

    data = self.payload.get("data", 'None')
    load = json.dumps(data)
    msg = load.encode('ascii')
    msg = struct.pack('>I', len(msg)) + msg
    self.sock.sendall(msg)
    self.sock.close()


def recv(sock):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    msg = recvall(sock, msglen).decode('ascii')
    data = json.loads(msg)
    payload = {'data': data}
    return payload


def recvall(sock, n=4):
    # Helper function to recv n bytes or return None if EOF is hit
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data += packet
    return data
