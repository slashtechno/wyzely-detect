# Wyze Face Recognition  
Recognize faces in Wyze Cam footage and send notifications to your phone (or other devices)  

## Pre-requisites  
* Docker  
* Docker Compose  
* A Wyze Cam  

## What's not needed  
* A Wyze Cam subscription  

## How to use
1. Clone this repo  
    ` git clone https://github.com/slackner/wyze-face-recognition.git`   
2. Add images to the `config` directory  
3. Copy `config/config.json.example` to `config/config.json` and edit the faces array to match the images you added, and the face names
4. Either set the `WYZE_EMAIL` and `WYZE_PASSWORD` environment variables, or edit `docker-compose.yml` and add your Wyze credentials  
5. Run `docker-compose up -d`  

### How to uninstall  
1. Run `docker-compose down` in the `wyze-face-recognition` directory  