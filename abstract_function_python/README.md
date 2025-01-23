# Flask BERT App - Docker

This is a simple Flask app that uses fine-tuned BERT models to classify pairs of states of web pages. The app is containerized using Docker.
Its main purpose is to provide an endpoint which the Crawljax Crawler can call to get an classification given two states represented by HTML.

### Building the image

Example:

```
docker build -t flask-classifier-app .
```

### Running the app

Example:

```
docker run -e FEATURE=content -e HF_MODEL_NAME=lgk03/NDD-claroline_test-content flask-classifier-app
```
