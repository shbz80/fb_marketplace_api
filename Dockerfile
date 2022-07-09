# install the base image with Python
FROM python:3.9-buster
# copy all contents of the project folder
COPY . .
# install all python package requirements
RUN pip install -r requirements.txt
# EXPOSE 8008
# runs the scraper program. amazon_book_scraper is a package
ENTRYPOINT ["python", "fb_mk_api.py"]
# the scraper program has two arguments: num_book and num_reviews
# both gets a default value of 5 and can be overdriven.
#CMD ["5", "5"]

# example docker run
# docker run -it -d -rm --name amazon_scraper shbz80/amazon_book_scraper:latest 10 10
# -it       : interactive 
# -d        : detached
# -rm       : remove after completion
# --name    : name of the container instance (amazon_scraper)
# [image]   : docker image (shbz80/amazon_book_scraper:latest)
# arg1, arg2: additional aguments to ENTRYPOINT (10 10)   
# other useful commands
# docker container ls   : list all active containers
# docker container ls -all  : list all containers
# docker container prune    : remove all stopped containers    