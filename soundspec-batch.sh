#!/usr/bin/env bash

if [ $# -ne 1 ]; then
  echo "Usage: $0 path_to_files"
  exit 1
fi

find "$1" -type f -print0 | while IFS= read -r -d '' afn; do
  echo; echo "${afn}"
  ffmpeg -loglevel warning -nostdin -i "${afn}" "${afn}".wav
  ./soundspec.py -b "${afn}".wav
  rm -f "${afn}".wav
done
