import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import pickle

#%matplotlib qt
# CALIBRATION
def CameraCalibrate(calDataPath):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    globalRet = False

    # Make a list of calibration images
    images = glob.glob(calDataPath +'/calibration*.jpg')
    
    # Step through the list and search for chessboard corners
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #print(gray.shape[::1])

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            globalRet = True

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
            cv2.imshow('img',img)
            cv2.waitKey(10)
    cv2.destroyAllWindows()
    
    if globalRet == True:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::1], None, None)
    else:
        ret = np.eye(3)
        dist = [[0, 0, 0, 0, 0]]
#    print(mtx)
#    print(dist)
    return mtx, dist


def cal_undistort(img, mtx,dist):
    undist =  cv2.undistort(img, mtx,dist, None, mtx)
    return undist

# READ and CALIBRATE IMAGE
def IMREAD(fname, mtx, dist):
     img = cv2.imread(fname)
     imgret = cal_undistort(img, mtx,dist);
     #cv2.imshow('img',img - imgret)
     #cv2.waitKey(1000)
     return imgret



# PERSPECTIVE CORRECTION
def warp_image(img,M):
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image
    return warped


def warper(src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    return M





def EstimateWrapParameter(mtx, dist,fTag):
    #dst = np.array([[200,720], [1000, 720], [1000, 200], [200, 200]], np.float32)
    #dst = np.array([[200,720], [1000, 720], [1000, 250], [200, 250]], np.float32)
    dst = np.array([[200,720], [1000, 720], [1000, 350], [200, 350]], np.float32)
    src = np.array([[200,720], [1150, 720], [750, 480], [550, 480]], np.float32)
    M =  warper(src, dst)
    Minv =  warper(dst,src)
    return M,Minv

def EstimateWrapParameterWrapper(mtx, dist, fromFile = False, updateToFile = True, fTag='straight_lines2*.jpg'):

    if os.path.isfile('WrapParameter.pkl') & fromFile: 
        pkl_file = open('WrapParameter.pkl', 'rb')
        M = pickle.load(pkl_file)
        Minv = pickle.load(pkl_file)
        pkl_file.close()
        print('TRUE')
    else:
        M, Minv = EstimateWrapParameter(mtx, dist,fTag)
        if updateToFile:
            output = open('WrapParameter.pkl', 'wb')
            pickle.dump(M, output)
            pickle.dump(Minv, output)
            output.close()
           
        print('FALSE')
    print(M)
    print(Minv)
    return M, Minv


# BINARIZATION CORRECTION
# Edit this function to create your own pipeline.
def BinarizeImage(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img2 = np.copy(img)
    img = np.copy(img)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    

    
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    #l_channel = hls[:,:,2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    
    ## USE ONLY R and G channel to replace B channel
    img2[:,:,2]= (img2[:,:,0]+img2[:,:,1])//2
    g = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY )
    mask = np.zeros_like(g)
    mask[g >= np.mean(g) ] = 1
    mask2 = np.zeros_like(g)
   
    mask2[g >= 50 ] = 1
    kernel = np.ones((7,7),np.uint8)
    mask2 = cv2.erode(mask2,kernel,iterations = 1)
    mask[mask2<1] = 0
    
    for i in range(3):
        color_binary[:,:,i] = color_binary[:,:,i]*mask

    return color_binary




# HISTOGRAM BASED LINE DETECTION
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin   # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        
         
        good_right_inds = ((nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        & (nonzeroy >= win_y_low) & (nonzeroy < win_y_high)).nonzero()[0]
        
        
        
        
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        ### TO-DO: If you found > minpix pixels, recenter next window ###
        ### (`right` or `leftx_current`) on their mean position ###
        #pass # Remove this when you add your function
    
        if len(good_left_inds) > minpix:
            leftx_current = np.int(  np.mean( nonzerox[good_left_inds] )  )    
    
        if len(good_right_inds) > minpix:
            rightx_current = np.int(  np.mean( nonzerox[good_right_inds] )  )    

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped, viz=False):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    ### TO-DO: Fit a second order polynomial to each using `np.polyfit` ###
    left_fit = np.polyfit(lefty,leftx,2);
    
    right_fit = np.polyfit(righty, rightx,2 );

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    if viz:
        # Plots the left and right polynomials on the lane lines
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')

    return out_img, left_fit, right_fit







def fit_poly(img_shape, leftx, lefty, rightx, righty, fit_order = 1):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    #left_fit = np.pol
    #right_fit = None
    left_fit = np.polyfit(lefty, leftx,fit_order);
    #left_fit = np.polyfit(lefty, leftx, 2)
    
    right_fit = np.polyfit(righty, rightx,fit_order );
    
   # right_fit = np.polyfit(righty, rightx, 2)
    
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = np.polyval(left_fit, ploty )
    right_fitx = np.polyval(right_fit, ploty )
    #left_fitx = None
    #right_fitx = None
    
    return left_fitx, right_fitx, ploty, left_fit, right_fit

def search_around_poly(binary_warped, left_fit, right_fit, viz=False):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100
    margin2 = 40
    

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    
    lfit = np.polyval(left_fit,nonzeroy)
    rfit = np.polyval(right_fit,nonzeroy)
    
    
    
    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = (nonzerox > lfit-margin) & (nonzerox < lfit+margin) 
    right_lane_inds = (nonzerox > rfit-margin) & (nonzerox < rfit+margin)
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty, left_fit_new, right_fit_new = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)
    
    
    
    lfit2 = np.polyval(left_fit_new,nonzeroy)
    rfit2 = np.polyval(right_fit_new,nonzeroy)
    
    
    
    left_lane_inds = (nonzerox > lfit-margin) & (nonzerox < lfit+margin) &  (nonzerox > lfit2-margin2) & (nonzerox < lfit2+margin2)
    right_lane_inds = (nonzerox > rfit-margin) & (nonzerox < rfit+margin) & (nonzerox > rfit2-margin2) & (nonzerox < rfit2+margin2)
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    
    left_fitx, right_fitx, ploty, left_fit_new_order2, right_fit_new_order2 = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty, 2)
    
    
    
    
    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                              ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                              ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    
    if viz:
        # Plot the polynomial lines onto the image
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        ## End visualization steps ##
    
    return result, left_fitx, right_fitx, ploty, left_fit_new, right_fit_new, left_fit_new_order2, right_fit_new_order2 



def generateNewFit(fit, y, M):
    
    
    x = np.polyval(fit, y)
    t = M[2,0]*x +  M[2,1]*y+ M[2,2]
    
    newX =  (M[0,0]*x +  M[0,1]*y+ M[0,2]) / t
    newY =  (M[1,0]*x +  M[1,1]*y+ M[1,2]) / t
    new_fit = np.polyfit(newY, newX,2)
    return new_fit
    
    
    
    
        
        
    
        

    



def measure_curvature_real(left_fit,right_fit, ploty, M):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    
    
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/765 # meters per pixel in x dimension
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = 719
    imageMid = 600
    
    
    
    
    left_fit_cr = np.copy(left_fit)
    #left_fit_cr = generateNewFit(left_fit, ploty, M)
    left_fit_cr[0] = left_fit_cr[0] * xm_per_pix/( ym_per_pix*ym_per_pix)
    left_fit_cr[1] = left_fit_cr[1] * xm_per_pix/ym_per_pix
    left_fit_cr[2] = left_fit_cr[2] * xm_per_pix

    right_fit_cr = np.copy(right_fit)
    #right_fit_cr = generateNewFit(right_fit, ploty, M)
    right_fit_cr[0] = right_fit_cr[0] * xm_per_pix/( ym_per_pix*ym_per_pix)
    right_fit_cr[1] = right_fit_cr[1] * xm_per_pix/ym_per_pix
    right_fit_cr[2] = right_fit_cr[2] * xm_per_pix
    
    
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    
    leftX =  np.polyval(left_fit,y_eval)
    rightX =  np.polyval(right_fit,y_eval)
    
    laneMid = (leftX+rightX)/2
    
    #delta = (laneMid-imageMid)
    
    delta = (laneMid - imageMid)*xm_per_pix
    
    return left_curverad, right_curverad, delta




def measure_curvature_real2(left_fit,right_fit):
    '''
    Calculates the curvature of polynomial functions in meters.
    '''
    
    
    
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    # Define y-value where we want radius of curvature
    # We'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval = 719
    imageMid = 600
    
    left_fit_cr = np.copy(left_fit)
    left_fit_cr[0] = left_fit_cr[0] * xm_per_pix/( ym_per_pix*ym_per_pix)
    left_fit_cr[1] = left_fit_cr[1] * xm_per_pix/ym_per_pix
    left_fit_cr[2] = left_fit_cr[2] * xm_per_pix
    
    
    
    
    right_fit_cr = np.copy(right_fit)
    right_fit_cr[0] = right_fit_cr[0] * xm_per_pix/( ym_per_pix*ym_per_pix)
    right_fit_cr[1] = right_fit_cr[1] * xm_per_pix/ym_per_pix
    right_fit_cr[2] = right_fit_cr[2] * xm_per_pix
    
    
    
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    
    leftX =  np.polyval(left_fit,y_eval)
    rightX =  np.polyval(right_fit,y_eval)
    
    laneMid = (leftX+rightX)/2
    
    #delta = (laneMid-imageMid)
    
    delta = (rightX - leftX) #*xm_per_pix
    
    return left_curverad, right_curverad, delta



if __name__ == "__main__":
    
    calDataPath = '../camera_cal'
    mtx, dist  = CameraCalibrate(calDataPath)
    #M = 
    Wrapper(mtx, dist)
    M = EstimateWrapParameterWrapper(mtx, dist, True, False)


