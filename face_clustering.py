from __future__ import division
import cv2
import face_recognition as fcr
import numpy as np
import imutils
from imutils.video import FileVideoStream
from imutils.video import FPS
import argparse
import time
import csv


def transform_to_grayscale(frame):
    # resize frame and transform to 3-channels grayscale
    frame = imutils.resize(frame, width=450)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.dstack([frame, frame, frame])
    return frame


def transform_to_rgb(frame):
    # resize frame and transfom to RGB
    frame = imutils.resize(frame, width=450)
    frame = frame[:, :, ::-1]
    return frame


def cluster_faces(known_faces_encodings_dict, face_encodings, tolerance=0.6):
    # compare face encodings to known faces encodings, if face encodings matches
    # with one of known faces encodings, then increase counter, else add face
    # encodings to known faces encodings
    face_indexes = list()
    known_faces_encodings = [d['encodings'] for d in known_faces_encodings_dict if 'encodings' in d]

    for face_encoding in face_encodings:
        index = -1
        matches = fcr.compare_faces(known_faces_encodings, face_encoding, tolerance)

        if True in matches:
            index = matches.index(True)
            result = known_faces_encodings_dict[index]
            result["counter"] += 1
        else:
            result = dict(encodings=face_encoding, counter=1)
            known_faces_encodings_dict.append(result)
            index = len(known_faces_encodings_dict) - 1

        face_indexes.append(index)

    return face_indexes


def process_video(video, tolerance=0.6, model="hog", upsample=1):
    # process video file and produce clustering results
    process_frame = True
    frames_with_face = 0
    known_faces_encodings = list()
    data = list()

    print("[INFO] Video name: " + video)
    print("[INFO] Starting video processing...")
    fvs = FileVideoStream(video).start()
    time.sleep(1.0)

    # division by two because every second frame is processed
    total_video_frames = int(fvs.stream.get(cv2.CAP_PROP_FRAME_COUNT)) / 2

    fps = FPS().start()

    while(fvs.more()):
        # read and transform the frame
        frame = transform_to_rgb(fvs.read())

        if (process_frame):
            timestamp = fvs.stream.get(cv2.CAP_PROP_POS_MSEC)
            frame_position = fvs.stream.get(cv2.CAP_PROP_POS_FRAMES)

            face_locations = fcr.face_locations(frame, upsample, model)
            face_encodings = fcr.face_encodings(frame, face_locations)
            face_indexes = cluster_faces(known_faces_encodings, face_encodings, tolerance)

            if (len(face_indexes) > 0):
                frames_with_face += 1

            result = dict(timestamp=timestamp, frame_position=frame_position,
                          face_indexes=face_indexes)
            data.append(result)

        process_frame = not process_frame

        # update the FPS counter
        fps.update()

    fps.stop()

    print("[INFO] Processing successfully finished")
    print("[INFO] Elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

    return total_video_frames, frames_with_face, known_faces_encodings, data


def calculate_face_importance(total_video_frames, frames_with_face, known_faces_encodings):
    results = list()
    for idx, d in enumerate(known_faces_encodings):
        overall_importance = d["counter"] / total_video_frames
        relative_importance = d["counter"] / frames_with_face
        result = [idx, overall_importance, relative_importance]
        results.append(result)

    return results


def export_to_csv(results, filename):
    with open(filename, 'wb') as csv_file:
        wr = csv.writer(csv_file, delimiter=',', quoting=csv.QUOTE_ALL)
        header = ["Face index", "Overall importance", "Relative importance"]
        wr.writerow(header)
        for row in results:
            wr.writerow(row)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face clustering for video file')
    parser.add_argument('-v', '--video', required=True, help="path to input video")
    parser.add_argument('-o', '--output', required=True, help="path to output results")
    # parameters
    parser.add_argument("-t", "--tolerance", type=float, default=0.6, help="max distance between compared faces")
    parser.add_argument("-m", "--model", default="hog", help="face detection model")
    parser.add_argument("-u", "--upsample", type=int, default=1, help=" how many times to upsample the image looking for faces")

    args = vars(parser.parse_args())

    total_video_frames, frames_with_face, known_faces_encodings, data = process_video(args["video"], args["tolerance"], args["model"], args["upsample"])
    results = calculate_face_importance(total_video_frames, frames_with_face, known_faces_encodings)
    export_to_csv(results, args["output"])
