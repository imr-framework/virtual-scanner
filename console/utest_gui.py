import unittest
from flask import Flask, url_for
from flask_testing import LiveServerTestCase
from selenium import webdriver

import virtualscanner.coms.coms_ui.coms_server_flask_alt as csfa

from virtualscanner.utils import constants
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait # available since 2.4.0
from selenium.webdriver.support import expected_conditions as EC # available since 2.26.0
from selenium.webdriver.common.by import By
import virtualscanner.coms.coms_ui.coms_server_flask as csf
import urllib.request
import sys


#app_address = {'win32': 'http://127.0.0.1:5000/',
 #              'maxOS': 'http://0.0.0.0:5000/',
  #             'linux': 'http://0.0.0.0:5000/'}


class TestGuiCase(LiveServerTestCase):
    def create_app(self):
        return csfa.create_app()

    def setUp(self):

        #self.driver = webdriver.Firefox(executable_path='./utils/drivers/geckodriver.exe')
        self.driver = webdriver.Chrome()
        self.driver.get(self.get_server_url())
        self.driver.get("https://127.0.0.1:5000/")

    def tearDown(self):
        self.driver.quit()

    def test_server_is_up_and_running(self):
        response = urllib.request.urlopen(self.get_server_url())
        #response = 200
        self.assertEqual(response, 200)



if __name__ == "__main__":
    unittest.main()

