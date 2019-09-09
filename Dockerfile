# Use an official Python runtime as a parent image
FROM python:3.6.3

# Add source code to /virtual-scanner
ADD . /virtual-scanner

# Set working directory to /virtual-scanner
WORKDIR /virtual-scanner

# Install requirements
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Expose Virtual Scanner's port 5000
EXPOSE 5000

CMD ["python", "virtualscanner/coms/coms_ui/coms_server_flask.py"]
