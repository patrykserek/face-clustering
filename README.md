# Face clustering
Clustering of faces appearing in the video.

## Requirements
* OpenCV -> [Ubuntu 16.04: How to install OpenCV](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)
* face_recognition -> [face_recognition Github page](https://github.com/ageitgey/face_recognition)
* imutils -> [imultis Github page](https://github.com/jrosebr1/imutils)

## Usage
Simple call:

`python face_clustering.py -v <video_path> -o <output_path>`

Call with parameters:

`python face_clustering.py -v <video_path> -o <output_path> -t <tolerance> -m <model> -u <upsample>`
