#!/bin/bash
# Define a timestamp function
timestamp() {
  date +"%T" # current time
}
while true; do
  echo -e "\n\n"
  timestamp 
  nvidia-smi --query-gpu=power.draw,utilization.gpu,utilization.memory --format=csv 
  sleep 1;
done >> 'temp.log'
