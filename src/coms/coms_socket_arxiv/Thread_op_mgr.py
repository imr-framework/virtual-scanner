"""
This script defines the Thread class and manages its functionality on run
Parameters
----------
    thread object

Performs
--------
   Tx to client
   Rx from client

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







import socket
import datetime
import coms_msg as msg
from threading import Thread

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
        if action == "Tx":
            serverlog.write(
                str(datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")) + ":Tx to: " + self.ip + " " + str(
                    self.port) + "\n")
            msg.send(self)  # Send message to client/server

        else:                #elif action == "Rx":
            serverlog.write(
                str(datetime.datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S")) + ":Rx from : " + self.ip + " " + str(
                    self.port) + "\n")
            payload = msg.recv(self.sock)  # Receive payload from client/server
            # process_payload(payload)


