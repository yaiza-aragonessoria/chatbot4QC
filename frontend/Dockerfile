FROM node:16-alpine as build

WORKDIR /app/frontend/

COPY package*.json /app/frontend/
RUN npm install

COPY . /app/frontend/

CMD ["npm", "build"]

FROM nginx:1.18.0-alpine

COPY ./webserver/default.conf /etc/nginx/conf.d/default.conf
COPY --from=build /frontend/build /usr/share/nginx/html