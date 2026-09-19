#!/bin/sh
set -e
cd "$(dirname "$0")"
OUT=out
ffmpeg -y -loglevel error -framerate 30 -i frames/f%05d.png \
  -c:v libx264 -preset slow -crf 17 -pix_fmt yuv420p -movflags +faststart \
  $OUT/flybrain_calculus_1080p.mp4
# square for social: PAD, do not crop. A 1080 crop slices the credit line
# mid-word and drops the question overlay, which is the hook.
ffmpeg -y -loglevel error -i $OUT/flybrain_calculus_1080p.mp4 \
  -vf "scale=1080:-2,pad=1080:1080:0:(oh-ih)/2:color=0x050710" \
  -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart \
  $OUT/flybrain_calculus_square.mp4
# looping silent webm, small enough to autoplay
ffmpeg -y -loglevel error -i $OUT/flybrain_calculus_1080p.mp4 \
  -vf "scale=1280:-2" -c:v libvpx-vp9 -b:v 1800k -an $OUT/flybrain_calculus_1280.webm
ffmpeg -y -loglevel error -i frames/f00520.png -vf scale=1920:-2 $OUT/poster.png
ls -lh $OUT/*.mp4 $OUT/*.webm $OUT/poster.png
