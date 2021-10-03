from PIL import Image
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
#import tensorflowjs as tfjs
import cv2  

classes = ({0:'angry',1:'disgust',2:'fear',3:'happy',
        4:'sad',5:'surprise',6:'neutral'})

classes_point = ({'angry':0, 'disgust':0, 'fear':0, 'happy':0, 'sad':0, 'surprise':0, 'neutral':0})

face_cascade_file = './haarcascade_frontalface_default.xml'  
front_face_detector = cv2.CascadeClassifier(face_cascade_file)  

cap = cv2.VideoCapture(0)
print(cap.isOpened())

model_path = './trained_models/fer2013_mini_XCEPTION.110-0.65.hdf5'
emotions_XCEPTION = load_model(model_path, compile=False)

minW = 0.1*cap.get(cv2.CAP_PROP_FRAME_WIDTH)  
minH = 0.1*cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  


while True:      
    tick = cv2.getTickCount()  

    ret, img = cap.read()
    #print(img)
    if(ret == False):
        continue
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = front_face_detector.detectMultiScale(   
        gray,  
        scaleFactor = 1.2,  
        minNeighbors = 3,  
        minSize = (int(minW), int(minH)),  
       )  
    
    for(x,y,w,h) in faces:  
         
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)  
        face_cut = img[y:y+h, x:x+w]
        face_cut = cv2.resize(face_cut,(64, 64))
        face_cut = cv2.cvtColor(face_cut, cv2.COLOR_BGR2GRAY)

        img_array = image.img_to_array(face_cut)
        pImg = np.expand_dims(img_array, axis=0) / 255

        prediction = emotions_XCEPTION.predict(pImg)[0]

        save_path = '../nodejs/static/emotion_XCEPTION'

        top_indices = prediction.argsort()[:][::-1]

        result = sorted([[classes[i], prediction[i]] for i in top_indices], reverse=True, key=lambda x: x[1])
        result_json = {classes[i]:prediction[i] for i in top_indices}
        #print(result)
        #print(result_json)

        classes_point['angry'] = classes_point['angry'] + result_json['angry']
        classes_point['disgust'] = classes_point['disgust'] + result_json['disgust']
        classes_point['fear'] = classes_point['fear'] + result_json['fear']
        classes_point['happy'] = classes_point['happy'] + result_json['happy']
        classes_point['sad'] = classes_point['sad'] + result_json['sad']
        classes_point['surprise'] = classes_point['surprise'] + result_json['surprise']
        classes_point['neutral'] = classes_point['neutral'] + result_json['neutral']
        
        
        result_c = result[0][0]
        

        '''
        result_p = [prediction[i] for i in top_indices]
        result_p_index = result_p.index(max(result_p))
        result_c = [classes[i] for i in top_indices][result_p_index]
        #print('=========')
        print(result_c)
        #print(result)
        #for x in result:
        #    print(x)
        '''
        cv2.putText(img, str(result_c), (x+5,y-5), cv2.FONT_HERSHEY_PLAIN, 5, (255,255,255), 2)
        
    # FPS算出と表示用テキスト作成  
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick)  
    # FPS  
    cv2.putText(img, "FPS:{} ".format(int(fps)),   
        (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2, cv2.LINE_AA)  
      
    cv2.imshow('camera',img)   

    # ESC  
    k = cv2.waitKey(10) & 0xff   
    if k == 27:
        print(classes_point)
        break  
 
print("\n Exit Program")  
cap.release()  
cv2.destroyAllWindows()  
