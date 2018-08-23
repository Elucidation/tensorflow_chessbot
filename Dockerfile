FROM tensorflow/tensorflow
MAINTAINER Sam <elucidation@gmail.com>

# Install python and pip and use pip to install the python reddit api PRAW
RUN apt-get -y update && apt-get install -y \
  python-dev \
  libxml2-dev \
  libxslt1-dev \
  libjpeg-dev \
  vim \
   && apt-get clean

# Install python reddit api related files
RUN pip install praw==4.3.0 beautifulsoup4==4.4.1 lxml==3.3.3 Pillow==4.0.0 html5lib==1.0b8

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Remove jupyter related files
RUN rm -rf /notebooks /run_jupyter.sh

# Copy code over
COPY . /tcb/

WORKDIR /tcb

# Run chessbot by default
CMD ["/tcb/run_chessbot.sh"]

# Start up the docker instance with the proper auth file using
# <machine>$ docker run -dt --rm --name cfb -v <local_auth_file>:/tcb/auth_config.py elucidation/tensorflow_chessbot
