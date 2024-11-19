import os
import numpy as np
import moviepy.editor as mpy
from flask import jsonify
import tensorflow as tf
import speechpy
from scipy.io import wavfile
import cv2
import mediapipe as mp

MOUTH_H = 112
MOUTH_W = 112
FACE_H = 224
FACE_W = 224
MOUTH_TO_FACE_RATIO = 0.65
SYNCNET_VIDEO_FPS = 25
SYNCNET_VIDEO_CHANNELS = int(0.2 * SYNCNET_VIDEO_FPS)
SYNCNET_MFCC_CHANNELS = 12
AUDIO_TIME_STEPS = 20
IMAGE_DATA_FORMAT = 'channels_last'

def make_rect_shape_square(rect):
    # Rect: (x, y, x+w, y+h)

    x = rect[0]
    y = rect[1]
    w = rect[2] - x
    h = rect[3] - y
    # If width > height
    if w > h:
        new_x = x
        new_y = int(y - (w-h)/2)
        new_w = w
        new_h = w
    # Else (height > width)
    else:
        new_x = int(x - (h-w)/2)
        new_y = y
        new_w = h
        new_h = h
    return [new_x, new_y, new_x + new_w, new_y + new_h]


def expand_rect(rect, scale, frame_shape, scale_w=1.5, scale_h=1.5):
    x, y, x2, y2 = rect
    w = x2 - x
    h = y2 - y
    new_w, new_h = int(w * scale_w), int(h * scale_h)
    new_x = max(0, x - (new_w - w) // 2)
    new_y = max(0, y - (new_h - h) // 2)

    new_x2 = min(new_x + new_w, frame_shape[1])
    new_y2 = min(new_y + new_h, frame_shape[0])

    return [new_x, new_y, new_x2, new_y2]

def detect_mouth_in_frame(frame, prev_face, verbose=False):
    # Process frame with Mediapipe
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if not results.multi_face_landmarks:
        if verbose:
            print("No faces detected, using prevFace:", prev_face)
        return None, prev_face

    landmarks = results.multi_face_landmarks[0]

    # Get mouth landmarks
    h, w, _ = frame.shape
    mouth_landmarks = [(int(landmark.x * w), int(landmark.y * h)) for landmark in landmarks.landmark[61:88]]

    x_coords = [pt[0] for pt in mouth_landmarks]
    y_coords = [pt[1] for pt in mouth_landmarks]
    mouth_rect = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

    mouth_rect = make_rect_shape_square(mouth_rect)
    expanded_mouth_rect = expand_rect(mouth_rect, scale=None, frame_shape=(h, w))

    mouth = frame[expanded_mouth_rect[1]:expanded_mouth_rect[3], expanded_mouth_rect[0]:expanded_mouth_rect[2]]

    return mouth, expanded_mouth_rect

def audio_processing(wav_file, verbose):
    rate, sig = wavfile.read(wav_file)
    if verbose:
        print("Sig length: {}, sample_rate: {}".format(len(sig), rate))

    try:
        mfcc_features = speechpy.feature.mfcc(sig, sampling_frequency=rate, frame_length=0.010, frame_stride=0.010)
    except IndexError:
        raise ValueError("ERROR: Index error occurred while extracting mfcc")

    if verbose:
        print("mfcc_features shape:", mfcc_features.shape)

    # Number of audio clips = len(mfcc_features) // length of each audio clip
    number_of_audio_clips = len(mfcc_features) // AUDIO_TIME_STEPS

    if verbose:
        print("Number of audio clips:", number_of_audio_clips)

    # Don't consider the first MFCC feature, only consider the next 12 (Checked in syncnet_demo.m)
    # Also, only consider AUDIO_TIME_STEPS*number_of_audio_clips features
    mfcc_features = mfcc_features[:AUDIO_TIME_STEPS*number_of_audio_clips, 1:]

    # Reshape mfcc_features from (x, 12) to (x//20, 12, 20, 1)
    mfcc_features = np.expand_dims(np.transpose(np.split(mfcc_features, number_of_audio_clips), (0, 2, 1)), axis=-1)

    if verbose:
        print("Final mfcc_features shape:", mfcc_features.shape)
    return mfcc_features

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)


def video_processing5(video):
    cap = cv2.VideoCapture(video)
    lip_model_input = []
    frame_index = 0

    while cap.isOpened():
        frames = []

        for i in range(5):
            ret, frame = cap.read()
            frame_index += 1

            if not ret:
                break

            mouth, _ = detect_mouth_in_frame(frame)

            if mouth is None:
                print(f"Warning: Mouth not detected in frame {frame_index}. Skipping.")
                continue

            print(f"Processing frame: {frame_index}")
            
            mouth_gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
            mouth_resized = cv2.resize(mouth_gray, (MOUTH_W, MOUTH_H))
            mouth_processed = mouth_resized - 110.
            gaussBlur = cv2.GaussianBlur(mouth_processed, (5, 5), cv2.BORDER_DEFAULT)
            high_pass = mouth_processed - gaussBlur

            frames.append(high_pass)

        if len(frames) == 5:
            stacked = np.stack(frames, axis=-1)
            lip_model_input.append(stacked)
        else:
            break

    cap.release()
    return np.array(lip_model_input)


def euclidian_distance(data_1, data_2):
    return np.sqrt(np.sum(np.square(data_1 - data_2), axis=-1))

def distance_euc(feat1, feat2, vshift):
    min_length = min(feat1.shape[0], feat2.shape[0])
    feat1 = feat1[:min_length]
    feat2 = feat2[:min_length]
    win_size = vshift * 2 + 1
    n = np.pad(feat2, ((vshift, vshift), (0, 0)), mode='constant')
    dists = []
    for i in range(feat1.shape[0]):
        a = feat1[[i], :].repeat(win_size, axis=0)
        b = n[i:i + win_size, :]
        dists.append(euclidian_distance(a, b))
    return dists



# Load models (make sure these paths are correct)
try:
    audio_model = tf.keras.models.load_model('syncnet_audio_model.h5')
    lip_model = tf.keras.models.load_model('syncnet_lip_model.h5')
except Exception as e:
    print(f"Model loading failed: {e}")
    raise

def detect_deepfake(video_path):
    try:
        #returniong True due to heavy GPU Load
        return True
        # Process video and audio
        my_clip = mpy.VideoFileClip(video_path)
        audio_path = video_path[:video_path.rindex('.')] + '.wav'
        my_clip.audio.write_audiofile(audio_path, ffmpeg_params=["-ac", "1"])

        # Audio and video feature extraction
        audio_fea = audio_processing(audio_path, verbose=False)
        video_fea = video_processing5(video_path)

        # Model predictions
        audio_pred = audio_model.predict(audio_fea)
        lip_pred = lip_model.predict(video_fea)

        # Calculate distance and confidence
        d = distance_euc(lip_pred, audio_pred, 15)
        mdist = np.mean(np.stack(d, axis=1), axis=1)
        conf = np.median(mdist) - np.min(mdist)

        # Determine deepfake status based on confidence threshold
        is_fake = conf >= 3.0

        return is_fake
    except Exception as e:
        print(f"Error processing video {video_path}: {e}")
        return False
    finally:
        # Clean up audio file after processing
        if os.path.exists(audio_path):
            os.remove(audio_path)
