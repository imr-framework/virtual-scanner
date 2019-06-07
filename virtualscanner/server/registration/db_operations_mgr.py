"""
This function inserts an entry into the database subject and table REGISTRATION
Parameters
----------
    payload

Performs
--------
    Creates database 'subject.db' with table 'REGISTRATION' if it does not exist locally
    Inserts the keys and values required for REGISTRATION table


Returns
-------
    status - 0: successfull 1: fail

Unit Test app
-------------
     utest_register
Author: Sairam Geethanath
Date: 03/07/2019
Version 0.0
Copyright of the Board of Trustees of  Columbia University in the City of New York
"""

import datetime
import sqlite3
from pathlib import Path

root = Path(__file__)
DB_PATH = root.parent / "subject.db"
DB_PATH = str(DB_PATH.resolve())
SERVERLOG_PATH = Path(__file__).parent / 'serverlog.txt'
SERVERLOG_PATH = SERVERLOG_PATH.resolve()


def create():
    status = 0  # successful unless caught by exception
    serverlog = open(SERVERLOG_PATH.resolve(), 'a')
    # Create the subject database
    serverlog.write("%s:Creating new database - subject.db\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    conn = sqlite3.connect(DB_PATH)

    # Create the REGISTRATION table
    try:
        conn.execute('''CREATE TABLE REGISTRATION
             (
             SUBJECTTYPE    TEXT    NOT NULL,
             PATID INT PRIMARY KEY     NOT NULL,
             NAME           TEXT    NOT NULL,
             AGE            INT     NOT NULL,
             DOB            BLOB    NOT NULL,
             GENDER         TEXT    NOT NULL,
             WEIGHT         REAL    NOT NULL,
             HEIGHT         REAL    NOT NULL,
             ORIENTATION    TEXT    NOT NULL,
             ANATOMY        TEXT    NOT NULL);''')
    except sqlite3.Error as e:
        serverlog.write(str(datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")) + ": " + str(e))
        status = 1
    if not status:
        serverlog.write("%s:Opened database successfully\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    conn.close()
    serverlog.close()
    return status


def insert(payload):
    serverlog = open(SERVERLOG_PATH, 'a')
    status = 0  # successful unless caught by exceptions
    try:
        conn = sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        serverlog.write(str(datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")) + ": " + str(e))
        status = 1
        print(e)

    serverlog.write("%s:Opened database successfully\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    cursor = conn.cursor()
    try:
        cursor.execute('INSERT INTO REGISTRATION VALUES (?,?,?,?,?,?,?,?,?,?)', list(payload.values()))
    except sqlite3.Error as e:
        serverlog.write(str(datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")) + ": " + str(e))
        status = 1
        print(e)

    try:
        conn.commit()
    except sqlite3.Error as e:
        serverlog.write(str(datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")) + ": " + str(e))
        status = 1
        print(e)

    serverlog.write(
        "%s:Inserted subject information to REGISTRATION table successfully\n" % datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"))
    conn.close()
    return (status)


def query(payload):
    serverlog = open(SERVERLOG_PATH, 'a')
    # supports only any 1 key at this time; need to extend this to multiple key value pairs
    status = 0  # successful unless caught by exceptions
    key = str(list(payload.keys()))
    key = key[2:-2]

    try:
        conn = sqlite3.connect(DB_PATH)
    except sqlite3.Error as e:
        serverlog.write("%s:Opened database successfully\n" % datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(e)

    cursor = conn.cursor()
    search = ("SELECT * FROM REGISTRATION WHERE " + key + "=?")

    try:
        cursor.execute(search, (list(payload.values())))
    except sqlite3.Error as e:
        serverlog.write(str(datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")) + ": " + str(e))
        status = 1
        print(e)
    rows = cursor.fetchall()
    return rows
