FROM node:16-alpine

WORKDIR /app/frontend

COPY ./package.json /app/frontend/
COPY ./package-lock.json /app/frontend/

RUN npm install

COPY . /app/frontend

#RUN npm run build
CMD [ "npm", "start" ]
