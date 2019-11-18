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



#subprocess.call(['python', str(constants.COMS_UI_PATH / 'coms_server_flask.py')], shell=True)

# Create a new instance of the Firefox driver


def get_vs_address():
    print(sys.platform)
    if sys.platform == 'win32':
        vs_address = "http://127.0.0.1:5000/"
    else:
        vs_address = "http://0.0.0.0:5000/"

    return vs_address

def setup():
    driver = webdriver.Firefox(executable_path="./firefox_driver/geckodriver.exe")
 # Go to Virtual Scanner
    driver.get(get_vs_address())
    return driver

def run_test(test, expected_value):

    driver = setup()

    value = test(driver)
    print("Expected value: ", expected_value)
    print("Test value: ", value)

    assert value == expected_value, "Value is not the same as expected :("
    print("")
    driver.quit()


def login_std(driver):
    # What does an implicit wait do?
    driver.find_element_by_id("mode-selection").submit()
    time.sleep(3)

    return driver.current_url


def login_adv(driver):
    driver.find_element_by_id("opt2").click()
    time.sleep(1)
    driver.find_element_by_id("mode-selection").submit()
    time.sleep(3)

    return driver.current_url


def register(driver):
    # Login first
    login_std(driver)

    # Find register button and click
    driver.find_element_by_id("reg-form-submit").click()
    time.sleep(3)

    # Check and return style of element with id="success-sentence"
    success_sentence = driver.find_element_by_id("success-sentence")
    time.sleep(3)

    return success_sentence.get_attribute("style")


def acquire(driver):
    register(driver)

    driver.find_element_by_link_text("Acquire").click()
    time.sleep(2)

    orientations = ['axial']
    seq_types = ['GRE']

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
           # driver.implicitly_wait(20)
            driver.find_element_by_css_selector(".submit-form-btn").click()

            element_found_result = 0
            # Wait until image is shown
            try:
                element = WebDriverWait(driver, 30).until(
#                    EC.presence_of_element_located((By.ID, "num0"))
                     EC.presence_of_element_located((By.XPATH,"/html/body/div/div/div/div/div/div/img[1]"))
                )
                print("Element found: ", element)
                element_found_result = "success"

            finally:
                driver.quit()
                result = element_found_result

            # If image is not shown after a while, raise error
            # Print success text
    return result

def analyze(driver):
    register(driver)
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

    t2_result = ""
    try:
        element = WebDriverWait(driver, 600).until(
            #                    EC.presence_of_element_located((By.ID, "num0"))
            EC.presence_of_element_located((By.XPATH, "/html/body/div/div/div/img[1]"))
        )
        t2_result = "success"

    finally:
        result = t2_result

    time.sleep(10)

    return result



def tx(driver):
    login_adv(driver)
    # How do you load a file?
    #C:\Users\tongg\Documents\Research\Code\virtual-scanner\virtualscanner\server\rf\tx\SAR_calc\assets

def rx(driver):
    login_adv(driver)

    driver.find_element_by_xpath("/html/body/div/div/nav/ul/li/a[@href=\"/rx\"]").click()
    time.sleep(1)
    input_dsf = driver.find_element_by_id("dsf")
    input_dsf.clear()
    input_dsf.send_keys("2")

    time.sleep(1)

    driver.find_element_by_id("form-submit-btn").click()
    result = ""
    try:
        element = WebDriverWait(driver, 20).until(
            #                    EC.presence_of_element_located((By.ID, "num0"))
            EC.presence_of_element_located((By.XPATH, "/html/body/div/div/div/div[contains(text(),'Resulting Image')]"))

        )
        result = "success"

    finally:
        return result

def launch_tests():
    vsadr = get_vs_address()
    webbrowser.open(vsadr)
    print(vsadr)
    # Run tests
    run_test(login_adv, vsadr + "recon")
    run_test(login_std, vsadr + "register")
    run_test(register, "display: block;")
    run_test(acquire, "success")
    #    run_test(analyze, "success") # This takes a long time (T2 mapping)
    run_test(rx, "success")
    webbrowser.close()


if __name__ == '__main__':
    launch_tests()



