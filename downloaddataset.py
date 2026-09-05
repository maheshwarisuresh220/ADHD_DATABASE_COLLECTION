import urllib.request

url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
output_name = "pose_landmarker.task"

print("Downloading MediaPipe model asset... Please wait.")
urllib.request.urlretrieve(url, output_name)
print("Download complete! 'pose_landmarker.task' is ready.")