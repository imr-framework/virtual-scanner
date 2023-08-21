from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait # available since 2.4.0
from selenium.webdriver.support import expected_conditions as EC # available since 2.26.0
from selenium.webdriver.common.by import By
import time
from virtualscanner.utils import constants
import subprocess
import sys
import unittest
import virtualscanner.coms.coms_ui.coms_server_flask_alt as csfa



#class TestPulseqGeneration(unittest.TestCase):
 #   def test_login(self):
# Launch virtualscanner
subprocess.call(['python', str(constants.COMS_UI_PATH / 'coms_server_flask.py')], shell=True)

# Create a new instance of the Firefox driver
driver = webdriver.Firefox(executable_path="./firefox_driver/geckodriver.exe")
# Go to Virtual Scanner
if sys.platform == 'win32':
    vs_address = "http://127.0.0.1:5000/"
else:
    vs_address = "http://0.0.0.0:5000/"

print(vs_address)
driver.get(vs_address)

"""
Log in
"""

driver.implicitly_wait(100)
begin = driver.find_element_by_id("mode-selection")
time.sleep(3)
begin.submit()

"""
Register
"""
driver.find_element_by_id("reg-form-submit").click()
time.sleep(5)

"""
Acquire
"""
driver.find_element_by_link_text("Acquire").click()
time.sleep(3)
# Acquire (5 x 5) axial, sagittal, and coronal images and check that they are displayed
orientations = ['axial','sagittal','coronal']
seq_types = ['Findf0', 'MonitorNoise', 'B0mapping', 'B1mapping', 'GRE', 'SE', 'IRSE' ]

for orientation in orientations:
    for seq_type in seq_types:

        ir = False
        if seq_type == 'IRSE':
            seq_name = 'SE'
            ir = False

        else:
            seq_name = seq_type


        # Add Sequence
        driver.find_element_by_id("addseq-btn").click()
        driver.find_element_by_link_text(seq_name).click()

        # Change parameters for faster sim (5 x 5)
    #    driver.implicitly_wait(20)
        inputnx = driver.find_element_by_id("Nx")
        inputnx.clear()
        inputnx.send_keys("5")
        driver.find_element_by_id("Ny").click()

        # Inversion recovery
      #  if ir:
      #      driver.find_element_by_id()

        # Simulate
       # driver.implicitly_wait(20)
        driver.find_element_by_css_selector(".submit-form-btn").click()

        # Wait until image is shown
      #  try:
       #     element = WebDriverWait(driver, 10).until(
        #        EC.presence_of_element_located((By.ID, "myDynamicElement"))
         #   )
        #finally:
         #   driver.quit()
        # If image is not shown after a while, raise error
        # Print success text


"""
Analyze
"""

#register = driver.find_element_by_id("reg-form-submit")
#register = driver.find_element_by_
#register.submit()



#try:
    # we have to wait for the page to refresh, the last thing that seems to be updated is the title
#    WebDriverWait(driver, 10).until(EC.title_contains("cats"))

    # You should see "cheese! - Google Search"
  #  print(driver.title)

#finally:
 #   driver.quit()

#WebDriverWait(driver, 100)
#driver.quit()


#if __name__ == '__main__':
 #   unittest.main()