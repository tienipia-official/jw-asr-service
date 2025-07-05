#!/bin/bash
docker build -t ghcr.io/tienipia/jw-asr-service:latest .
docker push ghcr.io/tienipia/jw-asr-service:latest