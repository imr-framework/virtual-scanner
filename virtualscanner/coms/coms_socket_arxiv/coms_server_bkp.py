"""
This script connects the server to multiple clients
Parameters
----------
    payload

Performs
--------
    Transmits (tx) payload to client(s)
    Receives (rx) payload from client(s)

Returns
-------
    Status of tx/rx

Unit Test app
-------------
     utest_coms
Author: Sairam Geethanath
Date: 03/11/2019
Version 0.0
"""

# server2.py
# The starter version of this code is from https://www.bogotobogo.com/python/python_network_programming_server_client_file_transfer.php

import socket
from threading import Thread
import datetime
import coms_msg as msg


# from SocketServer import ThreadingMixIn


def coms_server_exec(payload):
    TCP_IP = 'localhost'  # This will be replaced with STATIC IP later - one server many clients
    TCP_PORT = 9001
    BUFFER_SIZE = 1024  # Need to explore this further down the line for different payload types

    # Update serverlog
    serverlog = open(
        './serverlog.txt',
        'a')
    # Start coms server
    serverlog.write("%s:Running server coms\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    class ClientThread(Thread):

        def __init__(self, ip, port, sock, payload):  # payload already in JSON format
            Thread.__init__(self)
            self.ip = ip
            self.port = port
            self.sock = sock
            self.payload = payload

            serverlog.write(
                str(datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")) + " :New thread started for " + ip + ":" + str(
                    port) + "\n")

        def run(self):
            action = self.payload.get("action", 'None')
            if action == "Tx2client":
                serverlog.write(
                    str(datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S")) + ":tx to client: " + ip + " " + str(
                        port) + "\n")
                msg.send(self)  # Send message to client

            elif action == "Rxfromclient":
                print("Nothing")

            else:  # None case
                print("Nothing")

    tcpsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcpsock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcpsock.bind((TCP_IP, TCP_PORT))
    threads = []

    # Listening to requests from clients
    while True:  # Insert our two conditions here and get the run to work on action key of the payload
        tcpsock.listen(5)
        print("Waiting for incoming connections...")
        (conn, (ip, port)) = tcpsock.accept()
        print('Got connection from ', (ip, port))

        # Get payload from relevant source
        newthread = ClientThread(ip, port, conn, payload)
        # newthread.start()
        newthread.run()
        threads.append(newthread)

    # for t in threads:
    #     t.join()

# TODO: 1. Structure Tx2Client
#       2. Follow up with RxfromClient
#       3. Return payload and status
#       4. Shutdown before close

