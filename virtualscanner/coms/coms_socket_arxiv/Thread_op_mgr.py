"""
This script defines the Thread class and manages its functionality on run
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

if __name__ == '__main__':
    import os
    import sys

    script_path = os.path.abspath(__file__)
    SEARCH_PATH = script_path[:script_path.index('Virtual-Scanner') + len('Virtual-Scanner') + 1]
    sys.path.insert(0, SEARCH_PATH)

import datetime
from threading import Thread

import virtualscanner.coms.coms_socket_arxiv.coms_msg as msg

# Update serverlog
serverlog = open(
    './serverlog.txt',
    'a')


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
        print(action)
        if action == "tx":
            serverlog.write(
                str(datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")) + ":tx to: " + self.ip + " " + str(
                    self.port) + "\n")
            msg.send(self)  # Send message to client/server

        else:  # elif action == "rx":
            serverlog.write(
                str(datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")) + ":rx from : " + self.ip + " " + str(
                    self.port) + "\n")
            payload = msg.recv(self.sock)  # Receive payload from client/server
            # process_payload(payload)
