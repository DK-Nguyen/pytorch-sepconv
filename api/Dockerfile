# Start from the latest Long Term Support (LTS) Ubuntu version
FROM ubuntu:18.04

# Install pipenv
RUN apt-get update && apt-get install python3-pip -y && pip3 install pipenv

# Create the working directory
RUN set -ex && mkdir /repo
WORKDIR /repo