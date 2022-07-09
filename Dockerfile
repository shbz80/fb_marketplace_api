# install the base image with Python
FROM python:3.9-buster
# copy all contents of the project folder
COPY . .
# install all python package requirements
RUN pip install -r requirements.txt
# runs the api application program.
ENTRYPOINT ["python", "fb_mk_api.py"]

# example docker commands
# docker push shbz1980/fb_mk_api:latest 
# docker run -it -p 8008:8008 shbz1980/fb_mk_api:latest
# docker build -t shbz1980/fb_mk_api:latest .