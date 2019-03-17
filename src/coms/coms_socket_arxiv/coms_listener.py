"""
This script listens to client requests and invokes process payload method
Parameters
----------
    payload

Performs
--------
   Listens
   Rx payload
   Invokes process_payload

Returns
-------
    payload

Unit Test app
-------------
     utest_coms
Author: Sairam Geethanath
Date: 03/11/2019
Version 0.0
"""

# server2.py
# The starter version of this code is from https://www.bogotobogo.com/python/python_network_programming_server_client_file_transfer.php

# server2.py
import socket
from Thread_op_mgr import ClientThread
import coms_msg as msg

TCP_IP = 'localhost'
TCP_PORT = 9001
BUFFER_SIZE = 1024

tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
tcpsock.bind((TCP_IP, TCP_PORT))
threads = []

while True:
    tcpsock.listen(5)
    print("Waiting for incoming connections...")
    (conn, (ip, port)) = tcpsock.accept()
    print('Got connection from ', (ip, port))
    payload = msg.recv(conn)
    print("Rxed payload", payload)
    #Call process_payload(payload)


