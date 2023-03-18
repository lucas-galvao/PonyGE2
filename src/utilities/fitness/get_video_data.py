import cv2
import numpy as np
import tensorflow as tf
import random
import pathlib


def format_frames(frame, output_size):
    """
    Pad and resize an image from a video.

    Args:
      frame: Image that needs to resized and padded. 
      output_size: Pixel size of the output frame image.

    Return:
      Formatted frame with padding of specified output size.
    """
    frame = tf.image.convert_image_dtype(frame, tf.float32)
    frame = tf.image.resize_with_pad(frame, *output_size)
    return frame


def frames_from_video_file(video_path, n_frames, output_size = (224,224), format_frame = False):
    """
    Creates frames from each video file present for each category.

    Args:
      video_path: File path to the video.
      n_frames: Number of frames. 
      output_size: Pixel size of the output frame image.

    Return:
      An NumPy array of frames in the shape of (n_frames, height, width, channels).
    """
    result = []
    src = cv2.VideoCapture(str(video_path))  

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)
    frame_rate = src.get(cv2.CAP_PROP_FPS)
    frame_step = int(frame_rate * 0.5)
    need_length = 1 + (n_frames - 1) * frame_step
    
    if need_length > video_length:
        start = 0
    else:
        max_start = video_length - need_length
        start = random.randint(0, max_start + 1)
    
    src.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, frame = src.read()
    if format_frame:
        result.append(format_frames(frame, output_size))
    else:
        result.append(frame)

    for _ in range(n_frames - 1):
        for _ in range(frame_step):
            ret, frame = src.read()
        if ret:
            if format:
                frame = format_frames(frame, output_size)
            result.append(frame)
        else:
            result.append(np.zeros_like(result[0]))
    src.release()
    result = np.array(result)[..., [2, 1, 0]]

    return result


def get_files_and_class_names(path):
        video_paths = list(path.glob('*/*.mp4'))
        classes = [p.parent.name for p in video_paths] 
        return video_paths, classes


def get_video_frames(path_folder, n_frames, training = False, format_frame = False, output_size = (224, 224)):
    path_object = pathlib.Path(path_folder)
    video_paths, classes = get_files_and_class_names(path_object)
    class_names = sorted(set(p.name for p in path_object.iterdir() if p.is_dir()))
    class_ids_for_name = dict((name, idx) for idx, name in enumerate(class_names))

    pairs = list(zip(video_paths, classes))

    if training:
        random.shuffle(pairs)
    
    videos = []
    labels = []
    
    for path, name in pairs:
        video_frames = frames_from_video_file(path, n_frames, output_size, format_frame) 
        label = class_ids_for_name[name]
        videos.append(video_frames)
        labels.append(label)
    
    video_np = np.array(videos, dtype = float)
    labels_np = np.array(labels, dtype = int)

    return video_np, labels_np