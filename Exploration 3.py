#!/usr/bin/env python
# coding: utf-8

# # 3-1. 카메라 스티커앱 만들기 첫걸음

# 1. 얼굴인식 카메라의 흐름을 이해  
# 2. dlib 라이브러리 사용  
# 3. 이미지 배열의 인덱싱 예외 처리

# # 3-2. 어떻게 만들까? 사진 준비하기

# 눈, 코, 입, 귀(keypoint detection)와 같은 얼굴의 위치를 알아내기 위해 **랜드마크(landmark), 조정(alignment)** 기술 이용할 것

# 1. 얼굴이 포함된 사진을 준비하고  
# 2. 사진으로부터 얼굴 영역 face landmark 를 찾아냅니다. (landmark를 찾기 위해서는 얼굴의 bounding box를 먼저 찾아야합니다.)  
# 3. 찾아진 영역으로 부터 머리에 왕관 스티커를 붙여넣겠습니다.

# #### 디렉토리 구조 세팅

# mkdir -p ~/aiffel/camera_sticker/models  
# mkdir -p ~/aiffel/camera_sticker/images

# ##### 왕관 이미지 저장

# wget  
# https://aiffelstaticprd.blob.core.windows.net/media/original_images/king.png  
# 이미지 저장1, wget 명령어 사용  
# 
# wget  
# https://aiffelstaticprd.blob.core.windows.net/media/original_images/hero.png  
# 이미지 저장2  
# 
# mv king.png hero.png ~/aiffel/camera_sticker/images  
# 저장한 이미지 2개를 image 폴더로 이동시키기, mv 명령어 사용

# ##### 이미지 처리 패키지 다운

# pip install opencv-python  
# pip install cmake  
# pip install dlib
# 

# ##### 이미지 처리를 위해 opencv 와 노트북에 이미지를 출력하기 위한 matplotlib를 읽어오기

# In[187]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
print("🌫🛸")


# import를 해온다는거는 뭔가 도구를 가져온다고 이해하면 될듯하다.

# ##### 준비한 이미지 읽기

# In[188]:


import os
my_image_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/adf.jpeg'
img_bgr = cv2.imread(my_image_path)    #- OpenCV로 이미지를 읽어서
img_bgr = cv2.resize(img_bgr, (500, 400))    # 640x360의 크기로 Resize
img_show = img_bgr.copy()      #- 출력용 이미지 별도 보관
plt.imshow(img_bgr)
plt.show()


# 처음에 에러가 났는데 이유는 my_image_path에서 이미지 파일 이름을 코드와 다르게 적어놨다.

# 푸른빛이 나는 이유는 Opencv가 RGB가 아닌 BGR을 사용하기 때문이다. BGR을 사용하기 때문에 붉은색은 푸른색으로, 푸른색은 붉은색으로 바꿔 출력된다. 따라서 색깔 보정처리를 해줘야한다.

# ##### 색깔 보정처리

# In[189]:


# plt.imshow 이전에 RGB 이미지로 바꾸는 것을 잊지마세요. 
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()


# - 추가지식

# 이미지 읽기의 flag는 다음 3가지가 있습니다.   순서대로 1, 0, -1의 값을 갖습니다. 
# 
# 
# cv2.IMREAD_COLOR : 이미지 파일을 Color로 읽어들입니다. 투명한 부분은 무시되며, Default값입니다.
# cv2.IMREAD_GRAYSCALE : 이미지를 Grayscale로 읽어 들입니다. 실제 이미지 처리시 중간단계로 많이 사용합니다.
# cv2.IMREAD_UNCHANGED : 이미지파일을 alpha channel까지 포함하여 읽어 들입니다.
# 
# 
# cv2.imread('img.png', 0)이라고 호출했다면 이미지를 Grayscale로 읽어 들이겠다는 뜻입니다.

# # 3-3. 얼굴 검출 face detection

# ### Object detection 패키지 이용하기

# #### **dlib** 의 **face detector**는 **HOG(Histogram of Oriented Gradient) feature**를 사용해서 **SVM(Support Vector Machine)**의 **sliding window**로 얼굴을 찾습니다.

# https://medium.com/@jongdae.lim/%EA%B8%B0%EA%B3%84-%ED%95%99%EC%8A%B5-machine-learning-%EC%9D%80-%EC%A6%90%EA%B2%81%EB%8B%A4-part-4-63ed781eee3c

# **얼굴 인식은 일련의 여러가지 관련된 문제들을 해결하는 과정(Piepline)**

# 1. 먼저 사진을 보고 그 안에 있는 모든 얼굴을 찾아야 합니다.(Face detection)  
# 2. 둘째, 각 얼굴에 초점을 맞추고 얼굴이 이상한 방향으로 틀어졌거나 또는 조명이 안좋은 상황이라도, 여전히 같은 사람이라는 것을 이해할 수 있어야 합니다.
# 3. 셋째, 눈이 얼마나 큰지, 얼굴은 얼마나 긴지 등과 같이 다른 사람들과 구분하는데 사용하는 얼굴의 고유한 특징을 찾아 낼 수 있어야 합니다.
# 4. 마지막으로, 그 얼굴의 고유한 특징을 기존에 알고 있는 모든 사람들과 비교해서 그 사람의 이름을 결정해야 합니다.

# **Histogram of Oriented Gradients(HOG)**

# 1번 과정을 위해 이미지를 **흑백**으로 바꾼다. 얼굴 찾는데 색상 데이터가 필요 없기 때문이다.  
# 해당 픽셀이 이를 직접 둘러싸고 있는 픽셀들과 비교해서 **얼마나 어두운지** 알아내는 것이 목표이다.  
# 이미지가 **어두워지는 방향을 나타내는 화살표**를 그려야 한다.

# 이미지의 모든 픽셀에 대해 이 프로세스를 반복하면 결국 모든 픽셀이 **화살표**로 바뀌게 됩니다. 이러한 **화살표들을 그래디언트(gradients)**라고 부르고, 이를 통해 전체 이미지에서 **밝은 부분으로부터 어두운 부분으로의 흐름**을 알 수 있습니다:

# - 왜 굳이 그레디언트를 사용할까?  
# 
# 픽셀을 직접 분석하면, 동일한 사람의 정말 어두운 이미지와 정말 밝은 이미지는 전혀 다른 픽셀값을 갖게 될 것이다. 하지만 밝기가 변하는 방향만 고려하면 정말 어두운 이미지와 정말 밝은 이미지에 대한 완전히 동일한 표현(representation)을 얻게 됩니다.

# 다만, 모든 픽셀에 대해 그레디언트를 저장하면 너무 자세해지기 때문에 해당 이미지를 **16x16 픽셀의 정사각형**들로 분해합니다. 각 정사각형 안에서 **가장 우세한 그레디언트**를 표현하게 됩니다.

# 이렇게 얻은 HOG 이미지에서 얼굴을 찾기 위해서는 훈련 얼굴 이미지로부터 추출된 잘 알려진 HOG패턴과 가장 유사하게 보이는 부분을 test 이미지에서 찾는 것입니다.

# 우선, diib를 활용하여 hog detector를 선언

# In[190]:


import dlib
detector_hog = dlib.get_frontal_face_detector()   #- detector 선언
print("🌫🛸")


# Detector를 이용해서 얼굴의 bounding box를 추출

# In[191]:


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
dlib_rects = detector_hog(img_rgb, 1)   #- (image, num of img pyramid)
print("🌫🛸")


# - cvtColor() 를 이용해서 opencv 의 bgr 이미지를 rgb로 변환

# - image pyramids 이용하여 이미지를 upsampling하여 크기를 키운다. 얼굴을 다시 검출하면 작게 촬영된 얼굴을 크게 볼 수 있기 때문에 더 정확한 검출이 가능  
# https://opencv-python.readthedocs.io/en/latest/doc/14.imagePyramid/imagePyramid.html

# In[192]:


print(dlib_rects)   # 찾은 얼굴영역 좌표

for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()


# 이유는 모르겠으나 제가 사용하려던 파일들은 모두 얼굴 인식이 제대로 되지 않아 어쩔수 없이 예제 파일로 진행하겠습니다.

# # 3-4. 얼굴 랜드마크 face landmark

# 스티커를 섬세하게 적용하기 위해서는 **이목구비의 위치**를 아는 것이 중요. 이목구비의 위치를 추론하는 것을 **face landmark localization** 기술이라고 합니다. 
# 
# face landmark는 detection 의 결과물인 bounding box 로 잘라낸(crop) 얼굴 이미지를 이용합니다.

# #### Object keypoint estimation 알고리즘

# 객체 내부의 점을 찾는 기술을 object keypoint estimation이라 한다. key point를 찾는 알고리즘은 크게 2가지다.  
# 1. top-down :bounding box를 찾고 box 내부의 keypoint를 예측
# 2. bottom-up : 이미지 전체의 keypoint를 먼저 찾고 point 관계를 이용해 군집화 해서 box 생성  
# 
# 오늘은 1번 방식을 사용할겁니다.

# ### Dlib landmark localization

# ![image.png](attachment:image.png)

# 얼굴 이미지에서 68개의 이목구비 위치를 찾습니다. 점의 개수는 데이터셋과 논문마다 달라집니다.  예를 들면, AFLW 데이터셋은 21개를 사용하고 ibug 300w 데이터셋은 68개를 사용합니다.

# - Annotated Facial Landmarks in the Wild (AFLW)  
# - https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/

# #### ibug 300-W 데이터셋으로 학습한 Dlib를 사용하겠습니다.

# 우선 공개되어 있는 wegith file 다운

# $ wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# 다운 받은 파일을 옮겨줍니다.

# $ mv shape_predictor_68_face_landmarks.dat.bz2 ~/aiffel/camera_sticker/models

# cd를 이용하여 현재 위치를 바꿉니다.

# cd ~/aiffel/camera_sticker && bzip2 -d ./models/shape_predictor_68_face_landmarks.dat.bz2

# **저장한 landmark 모델 불러오기**

# In[193]:


import os
model_path = os.getenv('HOME')+'/aiffel/camera_sticker/models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)
print("🌫🛸")


# **landmark_predictor 는 RGB 이미지와 dlib.rectangle을 입력 받아 dlib.full_object_detection 를 반환합니다.**

# In[194]:


list_landmarks = []
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)

print(len(list_landmarks[0]))


# **랜드마크 출력해보기**

# In[195]:


for landmark in list_landmarks:
    for idx, point in enumerate(list_points):
        cv2.circle(img_show, point, 2, (0, 255, 255), -1) # yellow

img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()


# # 3-5. 스티커 적용하기

# **랜드마크를 기준**으로 '눈썹 위 얼굴 중앙' 에 스티커 붙이는 것이 목표!

# 얼굴 위치, 카메라의 거리에 따라 픽셀 x 가 다르기 때문에 비율로 계산을 해줘야한다. (x는 높이)

# 1. 스티커 위치  
# 2. 스티커 크기

# In[196]:


for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    print (landmark[30]) # nose center index : 30
    x = landmark[30][0]
    y = landmark[30][1] - dlib_rect.width()//2
    w = dlib_rect.width()
    h = dlib_rect.width()
    print ('(x,y) : (%d,%d)'%(x,y))
    print ('(w,h) : (%d,%d)'%(w,h))


# 이 경우 코의 중심점의 좌표가 (194, 149)  
# x는 landmark x좌표, y는 landmark y좌표에서 dlib_rect.width 절반해서 뺀 값  
# w,h는 스티커 크기

# Landmark [30]은 앞에서 봤던 랜드마크 점 중에서 30번째(0부터 시작했음)

# #### 스티커 이미지 읽기

# 여기서 위에 계산해둔 (187, 187)으로 resize하기

# In[197]:


import os
sticker_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/king.png'
img_sticker = cv2.imread(sticker_path)
img_sticker = cv2.resize(img_sticker, (w,h))
print (img_sticker.shape)


# 원본 이미지에 스티커 이미지를 추가하기 위해서 x, y 좌표를 조정합니다. 이미지 시작점은 top-left 좌표이기 때문입니다.

# In[198]:


refined_x = x - w // 2  # left
refined_y = y - h       # top
print ('(x,y) : (%d,%d)'%(refined_x, refined_y))


# 음수인 부분을 볼 수 있는데 원본 이미지를 벗어난 스티커 부분을 제거해야 하는 것을 의미.

# In[199]:


img_sticker = img_sticker[-refined_y:]
print (img_sticker.shape)


# top y 좌표를 원본 이미지의 경계 값으로 수정한다. 그래야지 벗어나는 부분이 사라지기 때문.

# In[200]:


refined_y = 0
print ('(x,y) : (%d,%d)'%(refined_x, refined_y))


# 이제 원본 이미지에 스티커를 적용하면 된다.

# In[201]:


sticker_area = img_show[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
img_show[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] =     np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)


# sticker_area는 원본이미지에서 스티커를 적용할 위치를 crop한 이미지 

# In[202]:


plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()


# 초록색 네모 박스(bounding box)와 여러 점들(landmark)을 제거 한다. 지금까지와 차이점이 있다면 img_show 대신 img_rgb를 사용할 것이다.

# In[203]:


sticker_area = img_bgr[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
img_bgr[refined_y:img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] =     np.where(img_sticker==0,sticker_area,img_sticker).astype(np.uint8)
plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
plt.show()


# # 3-6. 프로젝트: 고양이 수염 스티커 만들기

# #### Step 1. 고양이 수염 이미지/사진 파일을 준비합니다.

# #### Step 2. 얼굴 검출 & 랜드마크 검출 하기

# In[261]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
print("🌫🛸")


# In[262]:


import os
my_image_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/song'
img_bgr = cv2.imread(my_image_path)    #- OpenCV로 이미지를 읽어서
img_bgr = cv2.resize(img_bgr, (500, 400))    # 640x360의 크기로 Resize
img_show = img_bgr.copy()      #- 출력용 이미지 별도 보관
plt.imshow(img_bgr)
plt.show()


# In[263]:


# plt.imshow 이전에 RGB 이미지로 바꾸는 것을 잊지마세요. 
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()


# In[264]:


import dlib
detector_hog = dlib.get_frontal_face_detector()   #- detector 선언
print("🌫🛸")


# In[265]:


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
dlib_rects = detector_hog(img_rgb, 1)   #- (image, num of img pyramid)
print("🌫🛸")


# In[266]:


print(dlib_rects)   # 찾은 얼굴영역 좌표

for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()


# #### Step 3. 스티커 적용 위치 확인하기

# In[267]:


import os
model_path = os.getenv('HOME')+'/aiffel/camera_sticker/models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)
print("🌫🛸")


# In[268]:


list_landmarks = []
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)

print(len(list_landmarks[0]))


# In[269]:


for landmark in list_landmarks:
    for idx, point in enumerate(list_points):
        cv2.circle(img_show, point, 2, (0, 255, 255), -1) # yellow

img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()


# 여기까지는 왕관 씌울때와 동일하게 진행했다.

# #### Step 4. 스티커 적용하기

# In[270]:


for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    print (landmark[33]) # nose center index : 30
    x = landmark[33][0]
    y = landmark[33][1]
    w = dlib_rect.width()
    h = dlib_rect.width()
    print ('(x,y) : (%d,%d)'%(x,y))
    print ('(w,h) : (%d,%d)'%(w,h))


# 코 중심 좌표를 알게 됐으니 스티커 좌표도 거기에 맞추면 어떨까 생각했다.

# In[271]:


import os
sticker_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/cat-whiskers.png'
img_sticker = cv2.imread(sticker_path)
img_sticker = cv2.resize(img_sticker, (w,h))
print (img_sticker.shape)


# 스티커 저장 경로에서 스티커 파일을 불러오는 내용이다. 3번째 줄에서 스티커 사이즈를 w,h 사이즈(detection box)로 변경했다.

# In[272]:


refined_x = x - w // 2  # left
refined_y = y - h // 2     # top
print ('(x,y) : (%d,%d)'%(refined_x, refined_y))


# 이 부분을 알아내는데 시간이 굉장히 많이 걸렸다. 이미지의 (0, 0) 지점이 보통 생각하는 좌표평면의 원점이 아니고 top-left였다.따라서 위에 있는 왕관 예시에서 "-" 부분이 이미지의 위로 솟았던 것이다.  
# 
# 단순히 refined_x와 refined_y의 좌표를 코의 중심 좌표와 동일하게 만드는 것은 수염 스티커를 사람의 이마에 붙이는 결과를 낳았다. 이미지의 원점이 top-left에서 시작하는 것을 고려해서 좌표를 선정해야하는 것 같다.

# In[273]:


# img_sticker = img_sticker[-refined_y:]
print (img_sticker.shape)


# 왕관 씌우는 버전에서 refined_y = 0으로 만드는 코드가 있었는데 여기서는 그럴 필요가 없었다. 왜냐하면 애초에 refined_y가 양수값을 가지고 있으니 -refiend_y를 할 필요가 없고, 이미지 밖으로 나간 부분도 없으니 굳이 시작점을 0으로 할 필요도 없없다.

# In[274]:


img_show = img_bgr.copy()
sticker_area = img_show[refined_y:refined_y + img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
img_show[refined_y:refined_y + img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] =     np.where(img_sticker==0,img_sticker, sticker_area).astype(np.uint8)


# 이 부분에서는 2가지 문제가 있었다.  
# 
# 1. 스티커 좌표를 바꾸기 위해 위에서 수식을 바꾸면 이 부분에서 error가 발생했다. 고민하다가 팀원들에게 물어봤는데 img_show와 img_brg 부분을 수정하니 error가 발생하지 않았다. 코드를 자유롭게 변경할 수 있는 능력이 굉장히 중요하다고 생각한 부분이다. 파이썬 공부중인데 공부가 끝나면 좀더 자유롭게 코드 수정이 가능해졌으며 좋겠다.
# 
# 2. 수염 이미지가 어쩔때는 투명하고 어쩔때는 어두워져서 뭐가 문제인가 고민이었는데, np.where(img_sticker==0,img_sticker, sticker_area).astype(np.uint8)이 문제였다. 특히 img_sticker==0, img_sticker,sticker_area이 문제였다. 

# In[275]:


plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()


# ## 사실 이렇게만 보면 굉장히 간단해 보이지만 수염을 이마에서 코까지 내리는데 꼬박 일주일이 걸렸다... 분명히 내려와야 하는데 안내려올때마다 너무 고통스러웠다. 그냥 별 2개만 노리고 다른 과제는 안할까 생각했지만 하나는 해봐야겠다.  
# 
# ## 얼굴이 작은 이미지에는 어떻게 반응할지 살펴볼 예정입니다.

# In[1]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
print("🌫🛸")


# In[10]:


import os
my_image_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/abc.jpeg'
img_bgr = cv2.imread(my_image_path)    
img_bgr = cv2.resize(img_bgr, (800, 700))    
img_show = img_bgr.copy()      
plt.imshow(img_bgr)
plt.show()


# In[11]:


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb)
plt.show()


# In[12]:


import dlib
detector_hog = dlib.get_frontal_face_detector()   
print("🌫🛸")


# In[13]:


img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
dlib_rects = detector_hog(img_rgb, 1)  
print("🌫🛸")


# In[14]:


print(dlib_rects) 
for dlib_rect in dlib_rects:
    l = dlib_rect.left()
    t = dlib_rect.top()
    r = dlib_rect.right()
    b = dlib_rect.bottom()

    cv2.rectangle(img_show, (l,t), (r,b), (0,255,0), 2, lineType=cv2.LINE_AA)

img_show_rgb =  cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()


# ### (500,400)에서는 face detection이 안됐는데 그냥 (800,700)으로 바꿔보니 된다. 이미즈 사이즈가  detection 가능 여부에 영향을 주는거 같은데 이유는 아직 모르겠다.

# In[15]:


import os
model_path = os.getenv('HOME')+'/aiffel/camera_sticker/models/shape_predictor_68_face_landmarks.dat'
landmark_predictor = dlib.shape_predictor(model_path)
print("🌫🛸")


# In[16]:


list_landmarks = []
for dlib_rect in dlib_rects:
    points = landmark_predictor(img_rgb, dlib_rect)
    list_points = list(map(lambda p: (p.x, p.y), points.parts()))
    list_landmarks.append(list_points)

print(len(list_landmarks[0]))


# In[17]:


for landmark in list_landmarks:
    for idx, point in enumerate(list_points):
        cv2.circle(img_show, point, 2, (0, 255, 255), -1) # yellow

img_show_rgb = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
plt.imshow(img_show_rgb)
plt.show()


# In[18]:


for dlib_rect, landmark in zip(dlib_rects, list_landmarks):
    print (landmark[33])
    x = landmark[33][0]
    y = landmark[33][1]
    w = dlib_rect.width()
    h = dlib_rect.width()
    print ('(x,y) : (%d,%d)'%(x,y))
    print ('(w,h) : (%d,%d)'%(w,h))


# In[19]:


import os
sticker_path = os.getenv('HOME')+'/aiffel/camera_sticker/images/cat-whiskers.png'
img_sticker = cv2.imread(sticker_path)
img_sticker = cv2.resize(img_sticker, (w,h))
print (img_sticker.shape)


# In[20]:


refined_x = x - w // 2  # left
refined_y = y - h // 2     # top
print ('(x,y) : (%d,%d)'%(refined_x, refined_y))


# In[21]:


print (img_sticker.shape)


# In[22]:


img_show = img_bgr.copy()
sticker_area = img_show[refined_y:refined_y + img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]]
img_show[refined_y:refined_y + img_sticker.shape[0], refined_x:refined_x+img_sticker.shape[1]] =     np.where(img_sticker==0,img_sticker, sticker_area).astype(np.uint8)


# In[23]:


plt.imshow(cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB))
plt.show()


# ## 귀여운 스테판 커리가 탄생했다.
