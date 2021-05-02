import cv2

#Face and Eyes Detection in Real-Time
cap = cv2.VideoCapture(0)

#Videos are just a sequence of Images
#So, we will add a while loop to capture the frame continuously

faceCascade = cv2.CascadeClassifier("haarcascade_files/haarcascade_eye.xml")
faceCascade1 = cv2.CascadeClassifier("haarcascade_files/haarcascade_frontalface_default.xml")


while True:
    success, frame = cap.read() #frame variable will capture the Video & success variable will tell us whether it was captured successfully or not
            
        
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    eyes = faceCascade.detectMultiScale(imgGray, 1.1, 4)
    faces = faceCascade1.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,0),2)
    
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,0,0),2)
    
    cv2.imshow("Video", frame)
    
    if cv2.waitKey(1) == ord('q'): #This adds a Delay and looks for the key press inorder to break the loop
        break
            
        
cap.release() #Release the resources after recording
cv2.destroyAllWindows()