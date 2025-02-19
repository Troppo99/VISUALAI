import cv2

cap = cv2.VideoCapture(r'C:\xampp\htdocs\VISUALAI\assets\video\walking people.mp4')

subtractor = cv2.createBackgroundSubtractorMOG2(20, 50)

while True:
  ret, frame = cap.read()
  if ret:
    mask = subtractor.apply(frame)
    cv2.imshow('mask', mask)
    if cv2.waitKey(1) & 0xFF == ord('n'):
      break
  else:
    cap = cv2.VideoCapture(r'C:\xampp\htdocs\VISUALAI\assets\video\walking people.mp4')
cv2.destroyAllWindows()
cap.release()