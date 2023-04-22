#--------------------------------------------------------------------------------------------------------------------
# oakd_camera_stereo
# A sample python program that displays depth image from Oak-D stereo camera.
#
# Camera Specs
# Robotics Vision Core 2 (RVC2) with 16x SHAVE cores
#  -> Streaming Hybrid Architecture Vector Engine (SHAVE)
# Color camera sensor = 12MP (4032x3040 via ISP stream)
# Depth perception: baseline of 7.5cm
#  -> Ideal range: 70cm - 8m
#  -> MinZ: ~20cm (400P, extended), ~35cm (400P OR 800P, extended), ~70cm (800P)
#  -> MaxZ: ~15 meters with a variance of 10% (depth accuracy evaluation)
# https://docs.luxonis.com/projects/hardware/en/latest/pages/BW1098OAK.html
#
# Code
# The code in this file is based on the code from Luxonis Tutorials and Code Samples.
# https://docs.luxonis.com/projects/api/en/latest/tutorials/hello_world/
# https://docs.luxonis.com/projects/api/en/latest/tutorials/code_samples/
#
# Stereo specific
# https://docs.luxonis.com/projects/api/en/latest/tutorials/configuring-stereo-depth/
# https://docs.luxonis.com/projects/api/en/latest/samples/StereoDepth/depth_preview/#depth-preview
# https://docs.luxonis.com/projects/api/en/latest/samples/StereoDepth/stereo_depth_video/#stereo-depth-video
# https://docs.luxonis.com/projects/api/en/latest/samples/StereoDepth/depth_post_processing/#depth-post-processing
# https://docs.luxonis.com/projects/api/en/latest/samples/StereoDepth/rgb_depth_aligned/#rgb-depth-alignment
#
# Additional Info
# This website provides a good overview of the camera and how to use the NN pipeline.
# https://pyimagesearch.com/2022/12/19/oak-d-understanding-and-running-neural-network-inference-with-depthai-api/

#--------------------------------------------------------------------------------------------------------------------
# pip install numpy opencv-python depthai blobconverter
import numpy as np  # numpy package -> manipulate the packet data returned by depthai
import cv2  # opencv-python  package -> display the video stream
import depthai as dai  # depthai package -> access the camera and its data packets


#--------------------------------------------------------------------------------------------------------------------
# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False

# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = True

# Better handling for occlusions:
lr_check = True


#--------------------------------------------------------------------------------------------------------------------
# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight = pipeline.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

depth = pipeline.create(dai.node.StereoDepth)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("disparity")


#--------------------------------------------------------------------------------------------------------------------
# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)


#--------------------------------------------------------------------------------------------------------------------
# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xout.input)


#--------------------------------------------------------------------------------------------------------------------
# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    # Camera calibration data, which is need for focal length
    calibData = device.readCalibration()
    intrinsics = calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT)
    focal_pix = intrinsics[0][0]
    print('Right mono camera focal length in pixels:', focal_pix)

    # Stereo Camera baseline
    baseline_m = 0.075 # 7.5cm

    while True:
        #-------------------------------------------------
        # Get the disparity map/image
        # -------------------------------------------------
        inDisparity = q.get()  # blocking call, will wait until a new data has arrived
        disparityImg = inDisparity.getFrame()

        # TODO: There is a lot more work that needs to be done to get a "clean and continuous" disparity map.
        # The median filter is good for specular noise, but there are still noticeable noisy "regions" in the image.
        # https://docs.luxonis.com/projects/api/en/latest/samples/StereoDepth/depth_post_processing/

        # We need to eliminate any pixels with zero disparity, because they will cause an inf depth due to divide by zero.
        # clip < 1 will make the max range about 63 m
        # clip < 2 will make the max range about 31 m
        # clip < 3 will make the max range about 21 m
        # clip < 4 will make the max range about 15 m --> max effective range of OAK=D is 15m
        disparityImg = disparityImg.clip(4, None)

        # Convert from disparity to depth
        # https://docs.luxonis.com/projects/api/en/latest/tutorials/configuring-stereo-depth/
        # https: // docs.luxonis.com / projects / api / en / latest / samples / calibration / calibration_reader /  # camera-intrinsics
        # depth = baseline(m) * focal_length(pix) / disparity(pix)
        # note that if disparity = 0, then you get number/0.0 which is inf!!!
        depthImg = (baseline_m * focal_pix) / disparityImg



        # -------------------------------------------------
        # Visualization
        # -------------------------------------------------
        # Normalization image for better visualization
        disparityImg = (disparityImg * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

        # enhance visualization if configured for longer range
        if subpixel:
            disparityImg *= 2

        # Use OpenCV to display the image
        cv2.imshow("disparity", disparityImg)

        # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        disparityImg = cv2.applyColorMap(disparityImg, cv2.COLORMAP_JET)
        cv2.imshow("disparity_color", disparityImg)

        # depth image is float so you can't apply color map
        cv2.imshow("depth", depthImg)

        # -------------------------------------------------
        # Render images and check for q key
        # -------------------------------------------------
        if cv2.waitKey(1) == ord('q'):
            break