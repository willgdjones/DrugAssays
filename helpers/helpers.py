import numpy as np
import cv2

def normalize(image):
    image = np.array(image)
    max_r = max(image[:,:,0].flatten()) if max(image[:,:,0].flatten()) != 0 else 1
    max_g = max(image[:,:,1].flatten()) if max(image[:,:,1].flatten()) != 0 else 1
    max_b = max(image[:,:,2].flatten()) if max(image[:,:,2].flatten()) != 0 else 1

    n_image = np.zeros_like(image)
    n_image[:,:,0] = image[:,:,0] / max_r
    n_image[:,:,1] = image[:,:,1] / max_g
    n_image[:,:,2] = image[:,:,2] / max_b
    return n_image

def multiply_with_overflow(image, factor):
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

def random_chunk2(image, width, resize):
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
    
def random_chunk1(full_image, width):
    side = full_image.shape[0]
    x = np.random.randint(side - width)
    y = np.random.randint(side - width)
    return full_image[x:x+seg, y:y+seg,:]
