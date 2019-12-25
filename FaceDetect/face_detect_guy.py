import cv2
import sys
import numpy
import urllib.request

# Get user supplied values
url = sys.argv[1]
try:
    url_response = urllib.request.urlopen(url)
    img_array = numpy.array(bytearray(url_response.read()), dtype=numpy.uint8)
    image = cv2.imdecode(img_array, -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create the haar cascade
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    print(len(faces))
except:
    print('Image cannot be found or unreadable. Note: this script cannot read a local file, only an external URL.')
