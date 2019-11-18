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
# import virtualscanner.coms.coms_ui.coms_server_flask_alt as csfa
from selenium.webdriver.common.action_chains import ActionChains
import webbrowser

from virtualscanner.utils.constants import COMS_UI_PATH

#subprocess.call(['python', str(constants.COMS_UI_PATH / 'coms_server_flask.py')], shell=True)

# Create a new instance of the Firefox driver


class GUItestclass(unittest.TestCase):

    def get_vs_address(self):
        print(sys.platform)
        if sys.platform == 'win32':
            vs_address = "http://127.0.0.1:5000/"
        else:
            vs_address = "http://0.0.0.0:5000/"

        return vs_address

    def setup_driver(self):
        if sys.platform == 'win32':
            driver_path = str(COMS_UI_PATH / "firefox_driver/geckodriver_windows.exe")
        else:
            driver_path = str(COMS_UI_PATH / "firefox_driver/geckodriver_mac")
        print("Firefox driver path: ", driver_path)
        driver = webdriver.Firefox(executable_path=driver_path)
        # Go to Virtual Scanner

        return driver

    def run_test(self, test, expected_value):
        driver = self.setUp()
        value = test(driver)
        print("Expected value: ", expected_value)
        print("Test value: ", value)
        assert value == expected_value, "Value is not the same as expected :("
        driver.quit()

        return value==expected_value


    def login_std(self, driver):
        # What does an implicit wait do?
        driver.find_element_by_id("mode-selection").submit()
        time.sleep(3)
        result = driver.current_url
        return result


    def login_adv(self, driver):
        driver.find_element_by_id("opt2").click()
        time.sleep(1)
        driver.find_element_by_id("mode-selection").submit()
        time.sleep(3)
        result = driver.current_url
        return result


    def register(self, driver):
        # Login first
        self.login_std(driver)

        # Find register button and click
        driver.find_element_by_id("reg-form-submit").click()
        time.sleep(3)

        # Check and return style of element with id="success-sentence"
        success_sentence = driver.find_element_by_id("success-sentence")
        time.sleep(3)
        result = success_sentence.get_attribute("style") == "display: block;"
       # return success_sentence.get_attribute("style")
        return result


    def acquire(self, driver):
        self.register(driver)

        driver.find_element_by_link_text("Acquire").click()
        time.sleep(2)

        orientations = ['axial']
        seq_types = ['GRE']

        result = None

        for orientation in orientations:
            for seq_type in seq_types:
                ir = False
                if seq_type == 'IRSE':
                    seq_name = 'SE'
                    ir = True
                else:
                    seq_name = seq_type

                # Add sequence
                driver.find_element_by_id("addseq-btn").click() #
                time.sleep(1)
                driver.find_element_by_link_text(seq_name).click()
                time.sleep(1)

                # Change parameters for faster sim (5 x 5)
                inputnx = driver.find_element_by_id("Nx")
                inputnx.clear()
                inputnx.send_keys("5")
                driver.find_element_by_id("Ny").click()

                # Inversion recovery
                if ir:
                    driver.find_element_by_id("IRSE-check").click()

                # Simulate
                driver.find_element_by_css_selector(".submit-form-btn").click()

                element_found_result = False
                # Wait until image is shown
                try:
                    element = WebDriverWait(driver, 300).until(
                         EC.presence_of_element_located((By.XPATH,"/html/body/div/div/div/div/div/div/img[1]"))
                    )
                    element_found_result = element is not None

                finally:
                    result = element_found_result

                # If image is not shown after a while, raise error
                # Print success text
        return result

    def analyze(self, driver):
        self.register(driver)
        driver.find_element_by_link_text("Analyze").click()
        time.sleep(2)

        # T1 mapping
        driver.find_element(By.ID, "load-btn").click()
        element = driver.find_element(By.ID, "ui-id-4")
        actions = ActionChains(driver)
        actions.move_to_element(element).perform()
        time.sleep(1)
        t1 = driver.find_element(By.ID, "ui-id-6")
        actions.move_to_element(t1).click().perform()

        time.sleep(0.5)
        driver.find_element(By.CSS_SELECTOR, ".btn:nth-child(1)").click()

        t2_result = False
        try:
            element = WebDriverWait(driver, 600).until(
                EC.presence_of_element_located((By.XPATH, "/html/body/div/div/div/img[1]"))
            )
            t2_result = element is not None

        finally:
            result = t2_result

        time.sleep(3)
        return result


    def tx(self, driver):
        self.login_adv(driver)
        driver.find_element_by_xpath("/html/body/div/div/nav/ul/li/a[@href=\"/tx\"]").click()
        time.sleep(1)
        result = driver.current_url
        return result
        # How do you load a file?
        #C:\Users\tongg\Documents\Research\Code\virtual-scanner\virtualscanner\server\rf\tx\SAR_calc\assets

    def rx(self, driver):
        self.login_adv(driver)

        driver.find_element_by_xpath("/html/body/div/div/nav/ul/li/a[@href=\"/rx\"]").click()
        time.sleep(1)
        input_dsf = driver.find_element_by_id("dsf")
        input_dsf.clear()
        input_dsf.send_keys("2")

        time.sleep(1)

        driver.find_element_by_id("form-submit-btn").click()
        result = False
        try:
            element = WebDriverWait(driver, 20).until(
                #                    EC.presence_of_element_located((By.ID, "num0"))
                EC.presence_of_element_located((By.XPATH, "/html/body/div/div/div/div[contains(text(),'Resulting Image')]"))

            )
            result = element is not None

        finally:
            return result

    def test_login_std(self):
        driver1 = self.setup_driver()
        driver1.get(self.get_vs_address())
        self.assertEqual(self.login_std(driver1), self.get_vs_address()+"register")
        driver1.quit()

    def test_login_adv(self):
        driver2 = self.setup_driver()
        driver2.get(self.get_vs_address())
        self.assertEqual(self.login_adv(driver2),self.get_vs_address()+"recon")
        driver2.quit()

    def test_register(self):
        driver3 = self.setup_driver()
        driver3.get(self.get_vs_address())
        self.assertTrue(self.register(driver3))
        driver3.quit()

    def test_acquire(self):
        driver4 = self.setup_driver()
        driver4.get(self.get_vs_address())
        self.assertTrue(self.acquire(driver4))
        driver4.quit()


    def test_analyze(self):
        driver5 = self.setup_driver()
        driver5.get(self.get_vs_address())
        self.assertTrue(self.analyze(driver5))
        driver5.quit()

    def test_tx(self):
        driver6 = self.setup_driver()
        driver6.get(self.get_vs_address())
        self.assertEqual(self.tx(driver6),self.get_vs_address()+"tx")
        driver6.quit()

    def test_rx(self):
        driver7 = self.setup_driver()
        driver7.get(self.get_vs_address())
        self.assertTrue(self.rx(driver7))
        driver7.quit()



    # def launch_tests():
    #     vsadr = get_vs_address()
    #     tx_address = vsadr + "tx"
    #    # webbrowser.open(vsadr)
    #     print(vsadr)
    #     # Run tests
    #     run_test(login_adv, vsadr + "recon")
    #     run_test(login_std, vsadr + "register")
    #     run_test(tx, tx_address)
    #
    #     run_test(register, "display: block;")
    #     run_test(acquire, "success")
    #     #    run_test(analyze, "success") # This takes a long time (T2 mapping)
    #     run_test(rx, "success")
    #     #webbrowser.close()
    #
    #     return "Test Code"




if __name__ == '__main__':
    unittest.main()


