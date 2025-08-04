import cv2

def FrameCapture(path):
    vidObj = cv2.VideoCapture(path)
    count = 0
    success = 1
    while success:
        success, image = vidObj.read()
        cv2.imwrite("data/images/4/frame%d.jpg" % count, image)
        count += 1

if __name__ == '__main__':
    FrameCapture("data/video/VID_20250731_121122.mp4")
    print("Success!")