import cv2 
import numpy as np 
import os

# Fungsi untuk mendapatkan bilangan binary dari local patern
def get_pixel(img, center, x, y): 
	
	new_value = 0
	
	try: 
		# Jika pixel tetangga bernilai lebih besar dari nilai pixel centernya
		# maka pixel tetangga tersebut diubah menjadi nilai 1
		if img[x][y] >= center: 
			new_value = 1	
	except: 
		# Exception ketika pixel tetangga bernilai null
		pass
	
	return new_value 

# Fungsi untuk menghitung nilai baru untuk pixel center
def lbp_calculated_pixel(img, x, y): 

	center = img[x][y] 

	val_ar = [] 
	
	# top_left 
	val_ar.append(get_pixel(img, center, x-1, y-1)) 
	
	# top 
	val_ar.append(get_pixel(img, center, x-1, y)) 
	
	# top_right 
	val_ar.append(get_pixel(img, center, x-1, y + 1)) 
	
	# right 
	val_ar.append(get_pixel(img, center, x, y + 1)) 
	
	# bottom_right 
	val_ar.append(get_pixel(img, center, x + 1, y + 1)) 
	
	# bottom 
	val_ar.append(get_pixel(img, center, x + 1, y)) 
	
	# bottom_left 
	val_ar.append(get_pixel(img, center, x + 1, y-1)) 
	
	# left 
	val_ar.append(get_pixel(img, center, x, y-1)) 
	
	# Factor untuk menkonversi binary ke bilangan berbasis 10
	power_val = [1, 2, 4, 8, 16, 32, 64, 128] 

	val = 0
	
	for i in range(len(val_ar)): 
		val += val_ar[i] * power_val[i] 
		
	return val 

# Fungsi untuk mendapatkan gambar LBP
def lbp_implementation(path):
	img_path = path
	img_bgr = cv2.imread(img_path, 1) 

	height, width, _ = img_bgr.shape 

	# Mengubah gambar asli menjadi abu-abu
	img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) 

	# Membuat array bernilai 0 dengan ukuran yang sama dengan gambar asli
	img_lbp = np.zeros((height, width), np.uint8) 

	for i in range(0, height): 
		for j in range(0, width): 
			img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

	return img_lbp

def get_lbp_features(path):
	lbp_image = lbp_implementation(path).flatten()

	# Mean
	mean = np.mean(lbp_image)
	
	# Median
	median = np.median(lbp_image)
	
	# Standard Deviation
	std = np.std(lbp_image)
	n =  len(lbp_image)
	
	# Kurtosis
	squared_differences = (lbp_image - mean)**4	
	sum_of_squared_differences = np.sum(squared_differences)
	kurtosis = (4*sum_of_squared_differences) / (n*std**4) - 3
	
	# Skewness
	skewness = (3*(mean-median)) / std

	return [mean, median, std, kurtosis, skewness]

def get_lbp_name():
    return ['mean', 'median', 'std', 'kurtosis', 'skewness'] #5