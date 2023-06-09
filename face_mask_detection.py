import cv2
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

# Capture image from webcam
webcam = cv2.VideoCapture(0)

# Cascade object's face identifier
face_cascade = "haarcascade_frontalface_default.xml"
face_classifier = cv2.CascadeClassifier(face_cascade)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Define size of target face image
size = (224, 224)

while True:
  # Capture frame-by-frame
  success, image_bgr = webcam.read()
  
  image_bw = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
  
  # Convert BGR(cv2) to RGB
  image_rgb = cv2.cvtColor(image_bgr,cv2.COLOR_BGR2RGB)

  # Face detection using Cascade object's face identifier
  faces = face_classifier.detectMultiScale(image_bw)

  for face in faces:
    x, y, w, h = face
    
    # Convert array to image
    cface_rgb = Image.fromarray(image_rgb[y:y+h,x:x+w])

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1.
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = cface_rgb
    
    # resize the image to a 224x224 with the same strategy as in TM2:
    # resizing the image to be at least 224x224 and then cropping from the center
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    
    # log model prediction [[Masked, Non-Masked]] in 0 - 1 probability
    print(prediction)
    
    if prediction[0][0] > prediction[0][1]:
      cv2.putText(image_bgr,'Masked',(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),2)
      cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
      cv2.putText(image_bgr,'Non-Masked',(x,y-7),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2)
      cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0,0,255), 2)

  cv2.imshow("Mask Detection", image_bgr)
  
  # Hit 'q' on the keyboard to quit!
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release handle to the webcam
webcam.release()
cv2.destroyAllWindows()