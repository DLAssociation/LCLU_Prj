# Unfinished!
import gdal
import cv2
import numpy as np

def readTif(fileName):
    merge_img = 0
    driver = gdal.GetDriverByName('GTiff')
    driver.Register()

    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "掩膜失败，文件无法打开")
        return
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    print('im_width:', im_width)

    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    print('im_height:', im_height)
    im_bands = dataset.RasterCount  # 波段数
    im_geotrans = dataset.GetGeoTransform()  # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()  # 获取投影信息

    if im_bands == 1:
        band = dataset.GetRasterBand(1)
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
        cdata = im_data.astype(np.uint8)
        merge_img = cv2.merge([cdata, cdata, cdata])

        cv2.imwrite('a.jpg', merge_img)
    #
    elif im_bands == 4:
        # 	# im_data = dataset.ReadAsArray(0,0,im_width,im_height)#获取数据
        # 	# im_blueBand =  im_data[0,0:im_width,0:im_height] #获取蓝波段
        # 	# im_greenBand = im_data[1,0:im_width,0:im_height] #获取绿波段
        # 	# im_redBand =   im_data[2,0:im_width,0:im_height] #获取红波段
        # 	# # im_nirBand = im_data[3,0:im_width,0:im_height]   #获取近红外波段
        # 	# merge_img=cv2.merge([im_redBand,im_greenBand,im_blueBand])

        # 	# zeros = np.zeros([im_height,im_width],dtype = "uint8")

        # 	# data1 = im_redBand.ReadAsArray

        band1=dataset.GetRasterBand(1)
        band2=dataset.GetRasterBand(2)
        band3=dataset.GetRasterBand(3)
        band4=dataset.GetRasterBand(4)

        data1 = band1.ReadAsArray(0, 0, im_width, im_height).astype(np.uint16)  # r #获取数据
        data2 = band2.ReadAsArray(0, 0, im_width, im_height).astype(np.uint16)  # g #获取数据
        data3 = band3.ReadAsArray(0, 0, im_width, im_height).astype(np.uint16)  # b #获取数据
        data4 = band4.ReadAsArray(0, 0, im_width, im_height).astype(np.uint16)  # R #获取数据
        # 	print(data1[1][45])
        output1= cv2.convertScaleAbs(data1, alpha=(255.0/65535.0))
        # 	print(output1[1][45])
        output2= cv2.convertScaleAbs(data2, alpha=(255.0/65535.0))
        output3= cv2.convertScaleAbs(data3, alpha=(255.0/65535.0))

        merge_img1 = cv2.merge([output3, output2, output1])  # B G R

        cv2.imwrite('merge_img1.jpg', merge_img1)
        print('test')