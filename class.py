import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf


NEW_WIDTH=512
NEW_HEIGHT=512

#图片信息
def img_info(img):
    print(type(img))
    print(img.shape)
    print(img.size)  # 长*宽*通道数
    print(img.dtype)  # 编码格式
    pixel_data=np.array(img)#转换为array的像素信息
    print(pixel_data)


#视频展示
def video():
    capture=cv.VideoCapture()
    while True:
        ret,frame=capture.read()#frame为每一帧的画面
        cv.flip(frame,1)#左右颠倒变换
        cv.imshow('video',frame)
        c=cv.waitKey(50)
        if c==25:
            break

#图象转换
def img_convert(img):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)#灰度图
    #cv.imshow('gray',gray)
    hsv=cv.cvtColor(img,cv.COLOR_BGR2HSV)#hsv色彩空间
    lower_hsv=np.array([100,43,46])#蓝色在hsv色彩空间中的基本取值范围
    upper_hsv=np.array([124,255,255])
    mask=cv.inRange(hsv,lower_hsv,upper_hsv)#提取蓝色
    cv.imshow('mask',mask)
#    cv.imshow('hsv',hsv)


#基于tensorflow的图象大小转换
#return Tensor张量
'''
def resized(src):
    img_data=tf.image.decode_jpeg(src)
    img_data=tf.image.convert_image_dtype(img_data,dtype=tf.float32)
    resized=tf.image.resize_images(img_data,[256,256],method=0)
    return resized
'''

#基于opencv的图象大小调整
def resized(img):
    temp=img.copy()
    new_img=cv.resize(temp,(NEW_WIDTH,NEW_HEIGHT),interpolation=cv.INTER_AREA)
    return new_img



#调整图象的亮度和对比度
def contrast_lightness(img,c,b):#c为对比度增强倍数，b为亮度增强倍数
    h,w,channel=img.shape
    blank=np.zeros([h,w,channel],img.dtype)
    dst=cv.addWeighted(img,c,blank,1-c,b)
    return dst
#    cv.imshow('dst',dst)

#模糊操作去除噪声,卷积核已经被opencvAPI固定了
def median_blur(img):#中值模糊去掉椒盐噪声
    blur=cv.medianBlur(img,5)
    #cv.imshow('median_blur',blur)
    return blur

#自定义模糊(锐化)
def custom_blur(img):
    # kernel为自定义的算子,通过卷积核进行锐化，增强立体感(增强细节）
    kernel=np.array([[0,-1,0],[-1,5,1],[0,-1,0]],np.float32)
    dst=cv.filter2D(img,ddepth=-1,kernel=kernel)
    #cv.imshow('custom',dst)
    return dst

def clamp(pv):
    if pv>255:
        return 255
    if pv<0:
        return 0
    return pv


#手动增加高斯噪声
def gaussian_noise(img):
    h,w,c=img.shape
    for row in range(h):
        for col in range(w):
            s=np.random.normal(0,20,3)
            b=img[row,col,0]
            g=img[row,col,1]
            r=img[row,col,2]
            img[row, col, 0] = clamp(b + s[0])
            img[row, col, 1] = clamp(g + s[1])
            img[row, col, 2 ]= clamp(r + s[2])
    cv.imshow('gaussian',img)


#RGB通道分离
def RGB_split(img):
    b, g, r = cv.split(img)  # 通道的拆分
    cv.imshow('red',r)
    cv.imshow('green',g)
    cv.imshow('blue',b)


#RGB通道合并
def RGB_merge(b,g,r):
    new_img = cv.merge([b, g, r])  # 通道的合并
    new_img[:, :, 0] = 0  # 改变第三通道的颜色

#边缘保留滤波之高斯双边模糊
def bi_gaussian(img):#相当于加滤镜磨皮
    dst=cv.bilateralFilter(img,0,sigmaColor=100,sigmaSpace=15)#sigmaColor大点，小点
    cv.imshow('bi_gaussian',dst)


#边缘保留滤波之均值迁移
def mean_shift_gaussian(img):#效果更加模糊，相当于油画
    dst=cv.pyrMeanShiftFiltering(img,sp=10,sr=50)
    cv.imshow('mean_shift_gaussian',dst)


#直方图均衡化（基于灰度图）
#自动地增强图象的对比度，效果挺不错的
def equalHist(img):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    dst=cv.equalizeHist(gray)
    #cv.imshow('equalHist',dst)
    return dst


#局部直方图均衡化
#对比度限制的直方图均衡化,差异相对较小
def clahe(img):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    Clahe=cv.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    dst=Clahe.apply(gray)
    cv.imshow('clahe',dst)


#直方图反向投影
#可以用于在目标图片中找出样本图片,但是只能找出一样的图片，不能找出相关图片
def back_project():
    #样本图片
    sample=cv.imread('C:\\Users\\asus\\Desktop\\test\\number_plate.jpg')
    #目标图片
    target=cv.imread('C:\\Users\\asus\\Desktop\\test\\1.jpg')
    roi_hsv=cv.cvtColor(sample,cv.COLOR_BGR2HSV)
    target_hsv=cv.cvtColor(target,cv.COLOR_BGR2HSV)

    roiHist=cv.calcHist([roi_hsv],[0,1],None,[32,32],[0,180,0,256])#绘制直方图
    cv.normalize(roiHist,roiHist,0,255,cv.NORM_MINMAX)#归一化
    #直方图反向投影
    dst=cv.calcBackProject([target_hsv],[0,1],roiHist,[0,180,0,256],1)
    cv.imshow('back_project',dst)


#图象二值化(针对灰度图)
#全局阈值
def thresHold(img):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,binary=cv.threshold(gray,127,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    print('threshold rate is %s'%ret)
    cv.imshow('binary',binary)

#图象二值化之局部阈值
#这个效果比较好,但容易把图象搞花
def local_thresHold(img):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    binary=cv.adaptiveThresholdgray(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,15,0.1)#高斯阈值比平均阈值好
    cv.imshow('binary',binary)

#图象二值化之自定义阈值
def custom_thresHold(img):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    h,w=gray.shape[:2]
    m=np.reshape(gary,[1,h*w])
    mean=m.sum()/(h*w)
    print('mean :',mean)
    ret,binary=cv.threshold(gary,mean,255,cv.THRESH_BINARY)
    cv.imshow('binary',binary)


#超大图象二值化
def big_image_binary(img):
    print(img.shape)
    cw=256
    ch=256
    h,w=img.shape[:2]
    gary=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    for row in range(0,h,ch):
        for col in range(0,w,cw):
            roi=gary[row:row+ch,col:col+cw]#以255为一步切割
            print(np.std(roi),np.mean(roi))
            dev=np.std(roi)
            if dev<15:#空白图象过滤
                gary[row:row + ch, col:col + cw]=255
            else:
                ret, dst = cv.threshold(roi, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)#全局二值化
                gary[row:row + ch, col:col + cw] = dst
    cv.imwrite('big_image_binary.png',gary)
    cv.imshow('big_image_binary.png',gary)


#reduce图象金字塔
def pyramid(img):
    level=3
    temp=img.copy()
    pyramid_img=[]
    for i in range(level):
        dst=cv.pyrDown(temp)#下降
        pyramid_img.append(dst)
        cv.imshow('pyramid_down'+str(i),dst)
        temp=dst.copy()
    return pyramid_img

#由高斯金字塔得到拉普拉斯金字塔
#通过拉普拉斯金字塔可以得到图片的纹理
def lapalian(img):
    pyramid_img=pyramid(img)
    level=len(pyramid_img)
    for i in range(level-1,-1,-1):
        if (i-1)<0:
            expand=cv.pyrUp(pyramid_img[i],dstsize=img.shape[:2])#降采样
            lpls=cv.subtract(img,expand)
            cv.imshow('lapalian_down' + str(i), lpls)
        else:
            expand=cv.pyrUp(pyramid_img[i],dstsize=pyramid_img[i-1].shape[:2])
            lpls=cv.subtract(pyramid_img[i-1],expand)
            cv.imshow('lapalian_down' + str(i), lpls)


#一阶导数和sobel算子
#边缘提取
def sobel(img):
    #cv.Scharr是sobel的增强版本
    grad_x=cv.Sobel(img,cv.CV_32F,1,0)#x方向的sobel算子，#  确切的说是水平方向边缘检测的卷积核
    grad_y=cv.Sobel(img,cv.CV_32F,0,1)#y方向的sobel算子
    gradx=cv.convertScaleAbs(grad_x)#卷积操作
    grady=cv.convertScaleAbs(grad_y)
    #cv.imshow('gradx',gradx)
    #cv.imshow('grady',grady)

    gradxy=cv.addWeighted(gradx,0.5,grady,0.5,0)#alpha,beta都是比例系数
    cv.imshow('gradxy',gradxy)

#二阶导数和 拉普拉斯算子之opencvAPI
def lapalian_filter(img):
    dst=cv.Laplacian(img,cv.CV_32F)
    lpls=cv.convertScaleAbs(dst)
    cv.imshow('lapalian_filter',lpls)

#自定义拉普拉斯算子
#[[1,1,1],[1,-8,1],[1,1,1]]增强型
def custom_lapalian(img):
    kernel=np.array([[0,1,0],[0,-4,0],[0,1,0]])
    dst=cv.filter2D(img,cv.CV_32F,kernel=kernel)
    lpls=cv.convertScaleAbs(dst)
    cv.imshow('custom_lapalian',lpls)


#Canny边缘提取算法
def edge(img):
    #第一步高斯模糊，因为Canny算法对噪声很敏感
    blurred=cv.GaussianBlur(img,(3,3),0)
    #第二步灰度转换
    gray=cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)

    gray=cv.equalizeHist(gray)#补充一步直方图均衡化，增强对比度
    #第三步梯度检验
    gradx=cv.Sobel(gray,cv.CV_16SC1,1,0)#不能是浮点数
    grady=cv.Sobel(gray,cv.CV_16SC1,0,1)

    #求图象的边缘
    #edge_output=cv.Canny(gradx,grady,50,150,apertureSize=3)#低阈值50，高阈值150
    edge_output=cv.Canny(gray,50,150)
    cv.imshow('Canny_edge',edge_output)
    #输出彩色边缘
    dst=cv.bitwise_and(img,img,mask=edge_output)
    #cv.imshow('Color_edge',dst)
    return edge_output

#直线检测，霍夫直线变换
#这个算法有点问题
def line_detection(img):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges=cv.Canny(gray,50,150,apertureSize=3)#低阈值50，高阈值150
    #cv.imshow('Canny_edge',edges)
    lines=cv.HoughLines(edges,1,np.pi/180,200)
    for line in lines:
        rho,theta=line[0]
        a=np.cos(theta)
        b=np.sin(theta)
        x0=a*rho
        y0=b*rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)#(0,0,255)代表线的颜色

    cv.imshow('line_detection',img)

#第二种API,概率霍夫变换
#效果不太好
def line_detection_possible(img):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    edges=cv.Canny(gray,50,150,apertureSize=3)#低阈值50，高阈值150
    #cv.imshow('Canny_edge',edges)
    lines=cv.HoughLinesP(edges,1,np.pi/180,threshold=100,minLineLength=50,maxLineGap=10)
    for line in lines:
        x1,y1,x2,y2=line[0]
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # (0,0,255)代表线的颜色
    cv.imshow('line_detection_possible', img)


#轮廓发现
def contours(img):
    #先获取二值化后的图象
    #这是第一种做法,手动二值化
    dst=cv.GaussianBlur(img,(3,3),0)
    gray=cv.cvtColor(dst,cv.COLOR_BGR2GRAY)
    ret,binary=cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    #cv.imshow('binary',binary)
    #这是第二种做法
    #binary=edge(img)


    contours,hierarchy=cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        cv.drawContours(img,contours,i,(0,255,0),2)
        print(i)
    cv.imshow('contours',img)


#对象测量

def measure_object(img):
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,binary=cv.threshold(gray,0,255,cv.THRESH_OTSU|cv.THRESH_BINARY)
    #cv.imshow('binary',binary)
    dst = cv.cvtColor(binary,cv.COLOR_GRAY2BGR)
    contours,hireachy = cv.findContours(binary,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    for i,contour in enumerate(contours):
        area=cv.contourArea(contour)#轮廓的面积
        x,y,w,h=cv.boundingRect(contour)#轮廓的外接矩形，就是它外面那个框
        rate=max(w,h)/min(w,h)
        #print('the rectangle rate is %s'%rate)
        mm=cv.moments(contour)#求轮廓的几何矩
        if mm['m00']!=0.0:
            cx=mm['m10']/mm['m00']#中心的坐标
            cy=mm['m01']/mm['m00']
            cv.circle(img,(np.int(cx),np.int(cy)),3,(0,255,255),-1)
            cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)#矩形
            #多边形逼近
            approxCurve = cv.approxPolyDP(contour, 4, True)  #4是与阈值的间隔大小，越小越易找出，True是是否找闭合图像
            #print(approxCurve.shape)
            if approxCurve.shape[0] >= 7:
                cv.drawContours(dst, contours, i, (0, 255, 0), 2)# 画出轮廓
            if approxCurve.shape[0] ==4:
                cv.drawContours(dst, contours, i, (0, 0, 255), 2)
            else:
                cv.drawContours(dst, contours, i, (255, 0, 0), 2)

    cv.imshow('measure_object',dst)


#图象形态学操作
#腐蚀，原理与池化层的原理相同
#腐蚀能够让轮廓变得更加清晰
def erode(img):
    print(img.shape)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret,binary=cv.threshold(gray,0,255,cv.THRESH_OTSU|cv.THRESH_BINARY)
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(3,3))#卷积核3*3
    dst=cv.erode(binary,kernel=kernel)
    cv.imshow('erode',dst)

#膨胀
def dilate(img):
    print(img.shape)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #cv.imshow('gary',gray)
    ret,binary=cv.threshold(gray,0,255,cv.THRESH_OTSU|cv.THRESH_BINARY)
    #cv.imshow('binary',binary)
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(3,3))#卷积核3*3
    dst=cv.dilate(binary,kernel=kernel)
    cv.imshow('dilate',dst)


#形态学操作之开闭操作,应用于灰度图，二值图。开闭操作可以用于垂直或水平线提取
#开操作=腐蚀+膨胀，输入图象+结构操作。用于消除小的干扰区域。提取水平线
#闭操作=膨胀+腐蚀。输入图象+结构操作。用于填充闭合区域。提取数值线

def open(img):#开操作基本结构元素并未改变
    print(img.shape)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #cv.imshow('gary',gray)
    ret,binary=cv.threshold(gray,0,255,cv.THRESH_OTSU|cv.THRESH_BINARY)
    #cv.imshow('binary',binary)
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(3,3))#卷积核3*3效果比较好.当卷积核为(1,15)提取水平线
    binary=cv.morphologyEx(binary,cv.MORPH_OPEN,kernel)#执行形态学开操作

    cv.imshow('morph_open',binary)

#闭操作
def close(img):#会把小区域填充
    print(img.shape)
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #cv.imshow('gary',gray)
    ret,binary=cv.threshold(gray,0,255,cv.THRESH_OTSU|cv.THRESH_BINARY)
    #cv.imshow('binary',binary)
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(3,3))#卷积核3*3效果比较好。当卷积核为(15，1)提取水平线
    binary=cv.morphologyEx(binary,cv.MORPH_CLOSE,kernel)#执行形态学闭操作

    cv.imshow('morph_close',binary)


#顶帽 tophat  原图像和开操作之间的差值
#黑帽 blackhat 闭操作和原图像之间的差值
#形态学梯度 gradiant
#基本梯度 膨胀后的图象减去腐蚀后的图象的差值图象
#内部梯度 原图像减去腐蚀后的图象的差值图象
#外部梯度 膨胀后的图象减去原图像的差值图象

def top_hat(img):#效果真不错
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(15,15))
    dst=cv.morphologyEx(gray,cv.MORPH_TOPHAT,kernel=kernel)#形态学操作之顶帽
    #加点亮度
    cimage=np.array(gray.shape,np.uint8)
    cimage=50
    dst=cv.add(dst,cimage)

    cv.imshow('top_hat',dst)


def black_hat(img):#效果不太好
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(15,15))
    dst=cv.morphologyEx(gray,cv.MORPH_BLACKHAT,kernel=kernel)#形态学操作之黑帽
    #加点亮度
    cimage=np.array(gray.shape,np.uint8)
    cimage=50
    dst=cv.add(dst,cimage)

    cv.imshow('black_hat',dst)


#图象的形态学基本梯度
def gradient(img):#效果不太好,有点像提取轮廓
    gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dst=cv.morphologyEx(gray,cv.MORPH_GRADIENT,kernel=kernel)#形态学操作之黑帽
    #加点亮度
    cimage=np.array(gray.shape,np.uint8)
    cimage=50
    dst=cv.add(dst,cimage)

    cv.imshow('gradient',dst)

#图象的形态学内部梯度和外部梯度
def in_out_gradient(img):#效果不好
    kernel=cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    dm=cv.dilate(img,kernel=kernel)
    em=cv.erode(img,kernel=kernel)
    dst1=cv.subtract(img,em)#内部梯度 原图像减去腐蚀后的图象的差值图象
    dst2=cv.subtract(dm,img)#外部梯度 膨胀后的图象减去原图像的差值图象
    cv.imshow('internal',dst1)
    cv.imshow('external',dst2)


#基于距离变换的分水岭分割算法
'''算法流程
输入图象->灰度图->二值化->距离变换->寻找种子->生成maker->分水岭变换->输出图象
'''
def watershed(img):
    print(img.shape)
    blurred=cv.pyrMeanShiftFiltering(img,10,100)#去噪
    gray=cv.cvtColor(blurred,cv.COLOR_BGR2GRAY)
    ret,binary=cv.threshold(gray,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
    cv.imshow('binary',binary)

    kernel=cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    mb=cv.morphologyEx(binary,cv.MORPH_OPEN,kernel=kernel,iterations=2)#开操作
    sure_bg=cv.dilate(mb,kernel,iterations=3)#膨胀一下

    dist=cv.distanceTransform(mb,cv.DIST_L2,3)#距离变换
    dist_output=cv.normalize(dist,0,1.0,cv.NORM_MINMAX)
    #cv.imshow('dist_outout',dist_output*50)

    ret,surface=cv.threshold(dist,dist.max()*0.6,255,cv.THRESH_BINARY)#二值化
    #cv.imshow('surface_bin',surface)

    surface_fg=np.uint8(surface)
    unknown=cv.subtract(sure_bg,surface_fg)
    ret,markers=cv.connectedComponents(surface_fg)
    print(ret)

    markers=markers+1
    markers[unknown==255]=0
    markers=cv.watershed(img,markers=markers)
    img[markers==-1]=[0,0,255]

    cv.imshow('result_img',img)


url='C:\\Users\\asus\\Desktop\\test\\11.jpg'
img=cv.imread(url)  #img是numpy数组
#cv.imshow('input_img',img)

#blur=median_blur(img)#中值模糊去掉椒盐噪声
#cv.imshow('new',new_img)

#img_info(img)

#video()

#img_convert(blur)

#contrast_lightness(img,2,0)

#customBlur=custom_blur(img)
#img_convert(customBlur)

#gaussian_noise(img)
'''
#高斯模糊API,ksize为卷积核的大小,sigmaX为标准差
gaussian1=cv.GaussianBlur(img,(3,3),1)
#gaussian2=cv.GaussianBlur(img,(0,0),1)
cv.imshow('GaussianBlurAPI1',gaussian1)
#cv.imshow('GaussianBlurAPI2',gaussian2)
'''

#bi_gaussian(img)
#mean_shift_gaussian(img)

#equalHist(img)
#clahe(img)

#直方图反向投影
#back_project()

#二值化
#thresHold(img)
#local_thresHold(img)
#custom_thresHold(img)

#big_image_binary(img)

#图象大小调整
#new_img=resized(img)
#cv.imshow('new_input_img',new_img)

#图象金字塔
#reduce_pyramid=pyramid(new_img)
#lapalian(new_img)

#sobel算子和拉普拉斯算子
#sobel(img)
#lapalian_filter(img)
#custom_lapalian(img)

#Canny边缘检测算法
#edge(img)

#霍夫直线检测
#line_detection(img)
#line_detection_possible(img)

#轮廓发现
#contours(img)


#对象测量
#measure_object(new_img)

#形态学变换
#腐蚀
#erode(img)
#dilate(img)#膨胀

#开闭操作
#open(img)
#close(img)

#其他形态学操作
#顶帽 tophat
#top_hat(img)
#black_hat(img)#黑帽
#gradient(img)#基本梯度
#in_out_gradient(img)#内梯度和外梯度

#watershed(img)

cv.waitKey(0)
cv.destroyAllWindows()