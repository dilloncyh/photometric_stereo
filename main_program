import os, glob
import numpy as np
import scipy.linalg as spl
import matplotlib.pylab as plt
import cv2
import time

def load_files(path,extension,colour=False):
    """ This function loads all the image files with part of their filenames as
    'extension' in the directory 'path' in an array. If colour is 'True', then
    the images will have colour.""" 
    if colour == True:
        colour_setting = 1
    else:
        colour_setting = 0
    os.chdir(path)
    file_name_list =  glob.glob(extension)
    image_list = []
    name_list = []  ###for dubugging only
    
    for i in range(len(file_name_list)):
        image_list.append(cv2.imread(file_name_list[i], colour_setting))
        name_list.append(file_name_list[i])  ###for debugging only
    # print name_list  ### print the filenames in the array to make sure all sets of images are in the same order
    return np.array(image_list)
    
def ball_shape(img, min_radius=30, max_radius=50):
    """ This function requires images in an array and returns the coordinates
    of centre and radius of the circle detected. It is used to calculate (Cx,Cy)
    and radius r of the metal ball used for light calibration. It uses Hough
    Transform to detect the circle. The shape of metal ball should be easily
    distinguished (like putting a mobile phone behind the ball, and an estimate
    of the min and max value of the radius of the circle in units of pixel."""
    N = img.shape[0]
    circles = np.empty(shape=(N,3))  
    
    #check to see if the image is already in grayscale, and converts it if it is not
    for i in range(N):
        if img[i].shape[-1] == (3 or 4):
            img[i] = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)
            
        img[i] = cv2.medianBlur(img[i],5) # smoothen the image so no false circles are detected        
        circle_info = cv2.HoughCircles(img[i],cv2.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=min_radius,maxRadius=max_radius)

        if  circle_info == None:
            print i
            print 'Error, no circles found!'
            return None
        elif circle_info.shape[-1] != 3: # for checking
            print 'Error, more than one circle found!'
            print i
            print circle_info
            return None
        circles[i,:]  = circle_info     
              
    x = np.mean(circles[:,0])
    y = np.mean(circles[:,1])
    r = np.mean(circles[:,2])
    return (x, y, r)
    
def bright_spot(img, ROI=[150,300,150,250]): # if last is true, it is useful and checked as well 
    """This function locates the brightest spot on the metal ball. The region of
    interest (ROI) should be provided so that the brightest spot falls within this
    ROI and no false bright spot is detected."""
    N = img.shape[0]    
    top, bottom, left, right = ROI
    bright_spot_array = np.empty(shape=(N,2))
    
    # threshold the border of the image using the ROI 
    # so that no other brighter light source (e.g. mobile phone) is present 
    img2 = img.copy()
    img2[:,top:bottom,left:right] = 0
    img = img - img2
    
    for i in range(N):
        if img[i].shape[-1] == (3 or 4):
            img[i] = cv2.cvtColor(img[i], cv2.COLOR_BGR2GRAY)
        img[i] = cv2.GaussianBlur(img[i], (5,5), 0)
        bright_spot_array[i,:] = cv2.minMaxLoc(img[i])[3]
    return bright_spot_array
        
def masked_img(img,ROI,blur_radius=25,threshold=67):
    """This function requires a single image (image with all LEDs switched on)
    and returns an image that shows the regions masked by the object"""
    if img.shape[-1] == 3:
        img = img[:,:,-1]
        
    top,bottom,left,right = ROI    
    img2 = img.copy()
    img2[top:bottom,left:right] = 0
    img = img-img2
    img =  cv2.GaussianBlur(img, (blur_radius,blur_radius), 0) >  threshold 
    return img

def scale_image(img,white_board,background):
    """This function scales all the images by using the images of white board
    and background illumination."""
    
    if len(img.shape) == 3:
        N,height,width = img.shape
    elif len(img.shape) == 4:
        N,height,width,useless = img.shape
        img = img[:,:,:,2]
    else:
        print 'error! size of img array is not correct!'
        return None
    
    # check to see if the background img is in single colour channel (red)    
    background_light = np.empty(shape=(N,height,width))
    if len(background.shape) == 2:
        background_light[:] = background
    elif len(background.shape) == 3:
        background_light[:] = background[:,:,2]
        
    background_light[:] = cv2.GaussianBlur(background_light[:], (3,3), 0)
    
    # converts the data type of the array so that division results in decimals                 
    if img.dtype == 'uint8':
        img.astype(float)
    background.astype(float)
    white_board.astype(float)
    
    # scale the image
    img = (img-background_light)/(white_board-background_light)
    
    # converts all 'nan' resulted from 0/0 or 'inf' to 0
    img[img == np.nan] = 0
    img[img == float('inf')] = 0
    img[img == -float('inf')] = 0

    return img

def light_direction(ball_array, ball_mask_array, ball_ROI, R, D, far=True):
    """ This function computes the light direction vector L for each light LED
    using the img of metal ball.
    ball_array: array of img of metal ball
    ball_mask_array: the masked region of the metal ball
    ball_ROI: the region that includes the metal ball
    R: radius of the metal ball in cm
    D: distance from the object to the LEDs in cm
    far: True implies the distance can be treated as very far and approximations are made"""
    
    
    Xc, Yc, radius = ball_shape(ball_mask_array)
    bright_spot_array = bright_spot(ball_array, ball_ROI)
    D = D/(R/radius)
    
    # compute the no. of img, height and width of the img array
    if len(ball_array.shape) == 3: # image already in grayscale
        N, height, width = ball_array.shape
    else: # coloured image
        N, height, width, useless = ball_array.shape        

        
    bright_spot_normal = np.empty(shape=(N,3)) # for debugging   
    L_array = np.empty(shape=(N, height, width, 3))
    
    if far == False: # for correction of light direction at different points    
        x = np.arange(width)
        y = np.arange(height)
        index_map = np.meshgrid(x,y)
        
    for i in range(N):
        xo, yo = bright_spot_array[i]
        Nx = xo-Xc
        Ny = -(yo-Yc)
        Nz = np.sqrt(radius**2 - Nx**2 - Ny**2)
        n = np.array([Nx,Ny,Nz])/radius # surf normal vector at the bright spot
        print n
        L = 2*(n[2])*n - np.array([0,0,1])
        bright_spot_normal[i,:] = L
        
        # calculate the corrected L vector at different points if it is not 'far'
        # the method used here may not accurate as it only uses simple vector addition
        # more 'rigorous' expressions may be needed
        # all results presented in the project assumes 'far' = True
        if far == False:
            L_array[i,:,:,0] = D*(L[0]/L[2])-(index_map[0]-xo)
            L_array[i,:,:,1] = D*(L[1]/L[2])+(index_map[1]-yo)
            L_array[i,:,:,2] = D       
        else:
            L_array[i,:,:,:] = L
            
        # normalize the vector 
        L_array[i] = L_array[i] / np.sqrt((L_array[i]**2).sum(axis=-1))[:,:,None]
                 
    # L and L_array are all defined as the vector pointing from the object to the camera                
    return L_array, bright_spot_normal, D


def stacked_lstsq(L, b, rcond=1e-8):
    """
    Solve L x = b, via SVD least squares cutting of small singular values
    L is an array of shape (..., M, N) and b of shape (..., M).
    Returns x of shape (..., N)
    The algorithm is adopted from: http://stackoverflow.com/questions/30442377/how-to-solve-many-overdetermined-systems-of-linear-equations-using-vectorized-co
    This vectorized code is slightly faster than doing least-square for each pixel
    one-by-one, especially if L is different for different points
    """
    u, s, v = np.linalg.svd(L, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond*s_max
    inv_s = np.zeros_like(s)
    inv_s[s >= s_min] = 1/s[s>=s_min]
    x = np.einsum('...ji,...j->...i', v,
                  inv_s * np.einsum('...ji,...j->...i', u, b.conj()))
    return np.conj(x, x)


def surf_normal_test(img_array, img_mask, light_direction, blur=1, shadow=0.5):
    """this funcion calculates the surf normal of the object. 
    img_array: set of images of the actual object (in grayscale/single colour channel)
    img_mask: the masked region of the object (a 2D boolean array)
    light_direction: the array that stores the light direction for different images
    shadow: the threshold for detecting shadow. Set it to 0 to deactivate it.the L is Nx3 array of light direction.
    """
   
    N,height,width = img_array.shape
    total_pix = height*width
    img_mask.astype(float)
    img_array.astype(float)

    # smooth the image and calibrate the intensity
    if blur>0:
        img_array[:] = cv2.GaussianBlur(img_array[:], (blur,blur), 0)
    img_array = img_array*img_mask

    img_array = img_array.reshape(N,total_pix)
    
    L = np.empty(shape=(total_pix,N,3))
    b = np.empty(shape=(total_pix,N))
    
    for i in range(N):
        b[:,i] = img_array[i]
    

    light_direction = light_direction.reshape(N,total_pix,3) #checking
    #light_direction = light_direction.reshape(N,total_pix,3) #checking
    light_direction = np.rollaxis(light_direction,1) #checking
    L[:] = light_direction
    
    if shadow>0:
        b_median = np.median(b,axis=1)
        if N>=5:
            for pix in range(total_pix):
                for n in range(N):
                    if b[pix,n] < 0.3*b_median[pix]:
                        b[pix,n] = 0
                        L[pix,n,:] = 0

        elif N==4:
            b_min = np.argmin(b,axis=1)
            for pix in range(total_pix):
                if b[pix,b_min[pix]] < 0.5*b_median[pix]:
                    b[pix,b_min[pix]] = 0
                    L[pix,b_min[pix],:] = 0   
         
    surf_normal_map = stacked_lstsq(L,b).reshape(height,width,3)
    surf_normal_map = surf_normal_map/np.sqrt((surf_normal_map**2).sum(axis=-1))[:,:,None] #normalize the vector
    return surf_normal_map
        

def surf_height(surf_normal,penalty,estimate=0):
    """ this function reconstruct the surface using a regularized least-square
    method.
    surface_normal is the heightxwidthx3 array that stores the surface normal
    at each point
    penalty is the value of lambda"""
    
    
    def D_matrix(size):
        D_mat = np.zeros(shape=(size,size))
    
        D_mat[0,0] = -3
        D_mat[0,1] = 4
        D_mat[0,2] = -1
        D_mat[-1,-3] = 1
        D_mat[-1,-2] = -4
        D_mat[-1,3] = 3

        for row in range(1,size-1):
            D_mat[row,row-1] = -1
            D_mat[row,row+1] = 1
    
        D_mat = D_mat/2
        return D_mat
       
    normal_array[np.isnan(normal_array)]=0
    y,x = np.nonzero(surf_normal[:,:,0])
    left = np.min(x)
    right = np.max(x)
    top = np.min(y)
    bottom = np.max(y)
    surf_normal = surf_normal[top:bottom+1,left:right+1,:]
    #return surf_normal
    
    Zx = -surf_normal[:,:,0]
    Zy = surf_normal[:,:,1]

    height,width,useless = surf_normal.shape
    
    if estimate==0:
        guess=np.zeros(shape=(height,width))
     
    Dy = D_matrix(height)
    Dx = D_matrix(width)
    
    A = np.dot(Dy.T,Dy) + penalty**2*np.identity(height)
    B = np.dot(Dx.T,Dx) + penalty**2*np.identity(width)
    Q = np.dot(Dy.T,Zy )+ np.dot(Zx,Dx)+2*penalty**2*estimate
    
    Z = spl.solve_sylvester(A,B,Q)
    
    min_height = np.min(Z)
    Z = Z - min_height 
    
    Z = Z/masked_image[top:bottom+1,left:right+1]
    Z[Z==float('inf')] = np.nan
    Z[Z==(-float('inf'))] = np.nan

    return Z



### main program starts here ###
start_time = time.time()

path = "H:/Project/Data/15-6-Mon"


# load the images as arrays
# the first eight images are those that correspond to different light directions
# 9th image is when all lights are on
# 10th image is when all lights are off
ball_array = load_files(path,'*img_03*') # the image of the metal ball
ball_mask_array = load_files(path,'*img_04*') # the image of the metal ball with mobile phone behind
obj = load_files(path,'*img_05*',colour=True) # ping-pong ball (object)
board = load_files(path,'*img_02*',colour=True)[:-2,:,:,2] # image of the white board
board2 = load_files(path,'*img_01*',colour=True)[:-2,:,:,2] # image of the white board (take avg using two data)

board_avg = (board.astype(float)+board2.astype(float))/2 # take the avg value of the board

R = 5.0/2 # radius of the chrome ball in cm
D = 30.0 # distance from object to camera in cm
ball_ROI=[210,335,200,310] # region of interest for the ball mask [top,bottom,left,right]

far_setting = True

# light direction
L, bright_spot_normal, d = light_direction(ball_array[0:-2],ball_mask_array,ball_ROI,R,D,far=far_setting)

### white scattering ball as image
object_ROI = [170,335,185,350] #[top,bottom,left,right]
masked_image = masked_img(obj[-2],object_ROI)
background = obj[-1]
processed_image = scale_image(obj[0:-2],board_avg,background)
normal_array = surf_normal_test(processed_image,masked_image,L,blur=1)
height_map = surf_height(normal_array,0.03)


# plot the three components of the gradient field (surface normal)
top,bottom,left,right = object_ROI
component=['x-component','y-component','z-component']
for i in range(3): # turn the background into nan (no colour)
    normal_array[:,:,i]=normal_array[:,:,i]/masked_image     
for i in range(3): # plot the intensity graph
    plt.subplot(2,2,i+1)
    plt.imshow(normal_array[:,:,i])
    plt.xlim(left,right)
    plt.ylim(bottom,top)
    plt.axis('Off')
    plt.colorbar()
    plt.title(component[i])
    plt.show()

# plot the height map of the object
plt.subplot(2,2,4)
plt.title('height map')
plt.imshow(height_map)
plt.axis('Off')
plt.colorbar()
plt.show()

print time.time()-start_time
