FROM node:16-alpine as build

WORKDIR /app/frontend

COPY ./package.json /app/frontend/
COPY ./package-lock.json /app/frontend/

RUN npm install

COPY . /app/frontend

#RUN npm run build
CMD [ "npm", "start" ]

#FROM nginx:1.18.0-alpine
#
#COPY ./nginx/default.conf /etc/nginx/conf.d/default.conf
#COPY --from=build /app/frontend/build /usr/share/nginx/html
