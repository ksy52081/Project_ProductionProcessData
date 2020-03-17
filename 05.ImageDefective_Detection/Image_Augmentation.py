# importing libraries for basic tools 
import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np

#importing libraries for augmenting image files
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage import img_as_ubyte

#importing libraries for showing image files
import matplotlib.pyplot as plt

# finding location path of target images & location for saving image 폴더 안의 여러 이미지파일들을 가져옵니다.
targetimg_path = 'F:/JupyterLab/AdvancedIntelligence19/AI_Final/Data/train/train_def/'
saveimg_path = 'F:/JupyterLab/AdvancedIntelligence19/AI_Final/Data/Image_Augmented/'
#Datapath = "F:/JupyterLab/AdvancedIntelligence19/AI_Final/Data/Class1_def"
targetimg_list = os.listdir(targetimg_path)

for img_name in targetimg_list:
    # reading the image using its path
    targetimg_dir = targetimg_path+img_name
    saveimg_dir = saveimg_path+img_name
    
    # loading original image 오리지널 이미지 불러오기
    image = io.imread(targetimg_dir)
    
    #print('Rotated Image') ## 1-1. 90회전시키기
    rotated90 = rotate(image, angle=90, mode = 'wrap')  ## image 파일을 90 도만큼 회전시킵니다.
    #io.imshow(rotated90) ## plot the rotated image
    save_dir_rotated90 = saveimg_path+'rotated90_'+img_name ##이미지 디렉토리
    io.imsave(save_dir_rotated90, img_as_ubyte(rotated90))  ## 이미지 파일(array형태)저장하기
    
    #print('Rotated Image') ## 1-2. 180회전시키기
    rotated180 = rotate(image, angle=180, mode = 'wrap')  ## image 파일을 90 도만큼 회전시킵니다.
    #io.imshow(rotated90) ## plot the rotated image
    save_dir_rotated180 = saveimg_path+'rotated180_'+img_name ##이미지 디렉토리
    io.imsave(save_dir_rotated180, img_as_ubyte(rotated180))  ## 이미지 파일(array형태)저장하기
    
    #print('Rotated Image') ## 1-3. 270회전시키기
    rotated270 = rotate(image, angle=270, mode = 'wrap')  ## image 파일을 90 도만큼 회전시킵니다.
    #io.imshow(rotated90) ## plot the rotated image
    save_dir_rotated270 = saveimg_path+'rotated270_'+img_name ##이미지 디렉토리
    io.imsave(save_dir_rotated270, img_as_ubyte(rotated270))  ## 이미지 파일(array형태)저장하기

    #print('FlipedLR image') ## 2. 좌우 반전 이미지
    flipLR = np.fliplr(image)
    #io.imshow(flipLR) ## plot the fliped(LR) image
    #plt.title('Left to Right Flipped') ## plot the fliped(LR) image
    save_dir_flipedLR = saveimg_path+'flipedLR_'+img_name ##이미지 디렉토리
    io.imsave(save_dir_flipedLR, img_as_ubyte(flipLR)) ## 이미지 파일(array형태)저장하기
    
    #print('FlipedUD image') ## 3. 상하 반전 이미지
    flipUD = np.flipud(image)
    #io.imshow(flipUD) ## plot the fliped(UD) image
    #plt.title('Up Down Flipped') ## plot the fliped(UD) image
    save_dir_flipedUD = saveimg_path+'flipedUD_'+img_name ##이미지 디렉토리
    io.imsave(save_dir_flipedUD, img_as_ubyte(flipUD)) ## 이미지 파일(array형태)저장하기



    
    #print('Shift Image') ## 4. 수직수평이동
    transform = AffineTransform(translation=(25,25))
    image_wrapShift = warp(image,transform,mode='wrap')
    #io.imshow(wrapShift) ## plot the shifted image
    #plt.title('Wrap Shift') ## plot the shifted image
    save_dir_shifted = saveimg_path+'shifted_'+img_name ##이미지 디렉토리
    io.imsave(save_dir_shifted, img_as_ubyte(image_wrapShift)) ## 이미지 파일(array형태)저장하기
    
    #print('Noised Image') ## 5. 노이즈 추가 이미지
    sigma=0.155 #standard deviation for noise to be added in the image
    noisyRandom = random_noise(image,var=sigma**2)
    #io.imshow(noisyRandom)  ## plot the noised image
    #plt.title('Random Noise') ## plot the noised image
    save_dir_noised = saveimg_path+'noised_'+img_name ##이미지 디렉토리
    io.imsave(save_dir_noised, img_as_ubyte(noisyRandom)) ## 이미지 파일(array형태)저장하기
    
    #print('Blurred Image') ## 6. 희뿌연 이미지 
    blurred = gaussian(image,sigma=1,multichannel=True)
    #io.imshow(blurred) ##plot the blurred image
    #plt.title('Blurred Image') ##plot the blurred image
    save_dir_blurred = saveimg_path+'blurred_'+img_name ##이미지 디렉토리
    io.imsave(save_dir_blurred, img_as_ubyte(blurred)) ## 이미지 파일(array형태)저장하기


'''
io.imsave(saveimg_dir, rotated90) 를 그대로 쓰면 아래와 같은 에러가 발생합니다.
Lossy conversion from float64 to uint8. Range [0, 1]. Convert image to uint8 prior to saving to suppress this warning.
따라서 skimage에 있는 img_as_ubyte 함수를 사용해 형태를 바꿔줍니다
'''


'''
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.6000, 0.3946, 0.6041], [0.2124, 0.2335, 0.2360])
    ]),
'''