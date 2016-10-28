import numpy as np
import cv2
import os

class Assay():
    def __init__(self, ID, data_dir):
        self.ID = ID
        self.imageIDs = self.get_imageIDs(self.ID)
        self.label = 0
        self.data_dir = data_dir
        self.p_images = self.get_p_images(self.imageIDs, self.data_dir)
        
    def get_imageIDs(self, ID):
        prefixes = ['{}_{}'.format(ID,x) for x in ['s1','s2','s3','s4']]
        image_ids = [(x + '_w1.tif', x + '_w2.tif') for x in prefixes]
        return image_ids
    
    def get_p_images(self, images, data_dir):
         return [self.process_image(os.path.join(data_dir,image[0]),os.path.join(data_dir,image[1])) for image in images]


    def process_image(self,r_filepath, g_filepath):
        image1 = cv2.imread(r_filepath)
        image2 = cv2.imread(g_filepath)
        R = np.array(image1[:,:,0])
        G = np.array(image2[:,:,0])
        B = np.zeros(R.shape)
        raw_image = np.zeros([2160,2160,3])
        raw_image[:,:,0], raw_image[:,:,1], raw_image[:,:,2] = R, G, B
        n_image = self.normalize(raw_image)
        p_image = self.multiply_with_overflow(n_image, [3,3,1])
        return p_image
    
    def multiply_with_overflow(self,image, factor):
        m_image = np.zeros_like(image)

        m_imageR = cv2.multiply(image[:,:,0], factor[0])
        m_imageR[m_imageR > 1] = 1

        m_imageG = cv2.multiply(image[:,:,1], factor[1])
        m_imageG[m_imageG > 1] = 1

        m_imageB = cv2.multiply(image[:,:,2], factor[2])
        m_imageB[m_imageB > 1] = 1

        m_image[:,:,0] = m_imageR
        m_image[:,:,1] = m_imageG
        m_image[:,:,2] = m_imageB

        return m_image
    
    def normalize(self,image):
        image = np.array(image)
        max_r = max(image[:,:,0].flatten()) if max(image[:,:,0].flatten()) != 0 else 1
        max_g = max(image[:,:,1].flatten()) if max(image[:,:,1].flatten()) != 0 else 1
        max_b = max(image[:,:,2].flatten()) if max(image[:,:,2].flatten()) != 0 else 1

        n_image = np.zeros_like(image)
        n_image[:,:,0] = image[:,:,0] / max_r
        n_image[:,:,1] = image[:,:,1] / max_g
        n_image[:,:,2] = image[:,:,2] / max_b
        return n_image
    
    def random_chunk2(self, width, resize):
        X = width
        canvas = np.zeros(image.shape[0:2], dtype= "uint8")

        w = image.shape[0]
        r = int(np.floor(np.sqrt(2) * X))

        center_range = w - 2*np.ceil(r)

        x = np.random.randint(center_range)
        y = np.random.randint(center_range)

        C_mask = cv2.circle(canvas, (r + x,r + y), r, (255,255,255), -1)
        O = cv2.bitwise_and(image,image,mask=C_mask)

        theta = np.random.randint(360)

        M = cv2.getRotationMatrix2D((r + x,r + y), theta, 1.0)

        rotated = cv2.warpAffine(O, M, (w,w))

        chunk = rotated[y+r-X:y+r+X,x+r-X:x+r+X]
        resized_chunk = cv2.resize(chunk, (resize,resize), interpolation=cv2.INTER_CUBIC)

        di = np.random.randint(-1,3)

        if di == 2:
            return resized_chunk

        else:
            return cv2.flip(resized_chunk, di)

