import cv2
import numpy as np
import random
import os
# Load an image

for i in os.listdir("val2014"):
	print(i)
	file=i
	filename="val2014/" + i
	img = cv2.imread(filename)
	# Gaussian noise
	gaussian_noise = np.zeros(img.shape, np.float32)
	cv2.randn(gaussian_noise, mean=0, stddev=30)
	img_gaussian_noise = cv2.add(img.astype(np.float32), gaussian_noise)
	img_gaussian_noise = np.clip(img_gaussian_noise, 0, 255).astype(np.uint8)
	# Shot noise
	shot_noise = img.copy()
	shot_noise = np.array(shot_noise, np.float32)
	for i in range(img.shape[0]):
    		for j in range(img.shape[1]):
        		for k in range(img.shape[2]):
            			if random.random() < 0.1:
                			shot_noise[i][j][k] = 0
            			else:
                			shot_noise[i][j][k] = img[i][j][k]
	img_shot_noise = np.clip(shot_noise, 0, 255).astype(np.uint8)
	# Impulse noise
	impulse_noise = img.copy()
	impulse_noise = np.array(impulse_noise, np.float32)
	for i in range(img.shape[0]):
    		for j in range(img.shape[1]):
        		for k in range(img.shape[2]):
            			if random.random() < 0.1:
                			impulse_noise[i][j][k] = 255
            			else:
                			impulse_noise[i][j][k] = img[i][j][k]
	img_impulse_noise = np.clip(impulse_noise, 0, 255).astype(np.uint8)
	# Defocus blur
	defocus_blur = cv2.GaussianBlur(img, (21, 21), 0)
	img_defocus_blur = defocus_blur
	# Frosted Glass Blur
	frosted_glass_blur = cv2.GaussianBlur(img, (11, 11), 0)
	frosted_glass_blur = cv2.GaussianBlur(frosted_glass_blur, (11, 11), 0)
	img_frosted_glass_blur = img + img - frosted_glass_blur
	img_frosted_glass_blur = np.clip(img_frosted_glass_blur, 0, 255).astype(np.uint8)

	# Motion blur
	motion_blur_kernel = np.zeros((21, 21), np.float32)
	motion_blur_kernel[10,:] = 1.0/21
	motion_blur = cv2.filter2D(img, -1, motion_blur_kernel)
	img_motion_blur = motion_blur
	# Zoom blur
	zoom_blur_kernel = np.zeros((21, 21), np.float32)
	for i in range(21):
    		zoom_blur_kernel[i,:] = np.exp(-0.5*((i-10)/2)**2)
	zoom_blur_kernel = zoom_blur_kernel/np.sum(zoom_blur_kernel)
	zoom_blur = cv2.filter2D(img, -1, zoom_blur_kernel)
	img_zoom_blur = zoom_blur
	# Frost forms
	frost_forms = img.copy()
	frost_forms = np.array(frost_forms, np.float32)
	for i in range(img.shape[0]):
    		for j in range(img.shape[1]):
        		for k in range(img.shape[2]):
            			frost_forms[i][j][k] = frost_forms[i][j][k] + (random.random()-0.5)*30
	img_frost_forms = np.clip(frost_forms, 0, 255).astype(np.uint8)
	# Fog shrouds
	fog_shrouds = img.copy()
	fog_shrouds = np.array(fog_shrouds, np.float32)
	for i in range(img.shape[0]):
    		for j in range(img.shape[1]):
        		for k in range(img.shape[2]):
            			fog_shrouds[i][j][k] = fog_shrouds[i][j][k] - (random.random()-0.5)*30
	img_fog_shrouds = np.clip(fog_shrouds, 0, 255).astype(np.uint8)
	# Brightness
	img_brightness = cv2.addWeighted(img, 1.2, np.zeros(img.shape, img.dtype), 0, 10)
	img_brightness = np.clip(img_brightness, 0, 255).astype(np.uint8)
	# Contrast
	img_contrast = cv2.addWeighted(img, 1.5, np.zeros(img.shape, img.dtype), 0, -50)
	img_contrast = np.clip(img_contrast, 0, 255).astype(np.uint8)
	# Elastic transformations
	rows, cols = img.shape[:2]
	pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
	pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
	M = cv2.getAffineTransform(pts1, pts2)
	img_elastic_transformations = cv2.warpAffine(img, M, (cols, rows))
	# Pixelation
	img_pixelation = cv2.resize(img, (img.shape[1]//10, img.shape[0]//10),
                            interpolation=cv2.INTER_NEAREST)
	#img_pixelation = cv2.resize(img_pixelation, (img.shape[1], img.shape[0]),
	#                            interpolation=cv2.INTER_NEAREST)
	# JPEG Compression
	params = [cv2.IMWRITE_JPEG_QUALITY, 50]
	_, img_jpeg = cv2.imencode('.jpg', img, params)
	img_jpeg = cv2.imdecode(img_jpeg, cv2.IMREAD_COLOR)

	# Create a directory for the corrupted images
	corrupted_images_dir = 'corrupted_images'
	if not os.path.exists(corrupted_images_dir):
    		os.makedirs(corrupted_images_dir)
	i=file
	# Save each corrupted image
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_gaussian_noise.jpg'), img_gaussian_noise)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_shot_noise.jpg'), img_shot_noise)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_impulse_noise.jpg'), img_impulse_noise)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_defocus_blur.jpg'), img_defocus_blur)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_frosted_glass_blur.jpg'), img_frosted_glass_blur)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_motion_blur.jpg'), img_motion_blur)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_zoom_blur.jpg'), img_zoom_blur)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_frost_forms.jpg'), img_frost_forms)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_fog_shrouds.jpg'), img_fog_shrouds)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_brightness.jpg'), img_brightness)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_contrast.jpg'), img_contrast)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_elastic_transformations.jpg'), img_elastic_transformations)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_pixelation.jpg'), img_pixelation)
	cv2.imwrite(os.path.join(corrupted_images_dir,i[:-4]+'img_jpeg.jpg'), img_jpeg)



