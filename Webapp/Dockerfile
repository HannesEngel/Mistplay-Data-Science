FROM node:10


RUN apt update
RUN apt-get -y install python3-pip
 

RUN pip3 install --upgrade pip
RUN pip3 install pandas==0.23.4 scikit-learn==0.20.1 

WORKDIR /usr/src/app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000

RUN alias python=python3
CMD [ "node", "app.js" ]