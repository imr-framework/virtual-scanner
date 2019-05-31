"""
This script sends payload from server to client
Parameters
----------
    payload


Performs
--------
    tx to client


Returns
-------
    status

Unit Test app
-------------
     utest_coms
Author: Sairam Geethanath
Date: 03/11/2019
Version 0.0
"""

# The starter version of this code is from https://www.bogotobogo.com/python/python_network_programming_server_client_file_transfer.php
# client2.py
# !/usr/bin/env python
if __name__ == '__main__':
    import os
    import sys

    script_path = os.path.abspath(__file__)
    SEARCH_PATH = script_path[:script_path.index('Virtual-Scanner') + len('Virtual-Scanner') + 1]
    sys.path.insert(0, SEARCH_PATH)

import socket

from virtualscanner.coms.coms_socket_arxiv.Thread_op_mgr import ClientThread

TCP_IP = 'localhost'
TCP_PORT = 9001
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))


def exec(payload):
    newthread = ClientThread(TCP_IP, TCP_PORT, s, payload)
    newthread.run()
