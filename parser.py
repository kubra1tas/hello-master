import csv
from scipy import io
import numpy as np
import h5py as h5
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image as PIL_Image
import sys
import math
import scipy.io as sio
import pyquaternion as pq
import transformation
import time
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import datetime
from random import randrange
import rosbag
from std_msgs.msg import Int32, String

filename = '/home/kub/Downloads/rosbag_converter/step_and_shoot.bag'
rosbag_dir = "/home/kub/Downloads/rosbag_converter/step_and_shoot.bag"

# sensor_msgs::JointState
t_rob6_joints = "/rob6_joints"
# rt_msgs::TransformRTStampedWithHeader
t_rob6_tf = "/rob6_tf"
# sensor_msgs::Image
# t_pico_gray = "/royale_camera_driver/gray_image"
t_pico_gray = "/royale_camera_driver/gray_image"
# ::Image
# t_pico_depth = "/royale_camera_driver/depth_image"
t_pico_depth = "/royale_camera_driver/depth_image"
# sensor_msgs::PointCloud2
t_pico_pointcloud = "/royale_camera_driver/point_cloud"
# rt_msgs::TransformRTStampedWithHeader
t_atracsys_head = "/trk_head_tf"
# rt_msgs::TransformRTStampedWithHeader
t_atracsys_coil = "/trk_coil_tf"

def save_topics(group, msg_topic, msg):
	debug = False
	width = 224
	height = 171
	if msg_topic == t_rob6_joints:
		rob6_joints = np.zeros(6)
		if debug:
			print(msg_topic)
			print(msg.position)
		tf = []
		for i in msg.position:
			if math.isnan(i):
				print("Nan detected", i, msg.position)
			tf.append(i)
		rob6_joints = np.asarray(tf)
		group.create_dataset("rob6_joints", data=rob6_joints, compression="gzip", compression_opts=9)

	if msg_topic == t_rob6_tf:
		rob6_tf = np.zeros(7)
		tf = []
		# for i in msg.transform.translation:
		#    tf.append(i)
		tf.append(msg.transform.translation.x)
		tf.append(msg.transform.translation.y)
		tf.append(msg.transform.translation.z)
		tf.append(msg.transform.rotation.x)
		tf.append(msg.transform.rotation.y)
		tf.append(msg.transform.rotation.z)
		tf.append(msg.transform.rotation.w)

		# for j in msg.transform.rotation:
		#    tf.append(j)

		rob6_tf = np.asarray(tf)
		group.create_dataset("rob6_tf", data=rob6_tf, compression="gzip", compression_opts=9)
		if debug:
			print(msg_topic)
			print("tf msg:", msg.transform.translation, msg.transform.rotation)

	if msg_topic == t_pico_gray:
		np.set_printoptions(threshold=sys.maxsize)
		np_arr = np.fromstring(msg.data, np.uint16)

		if np.isnan(np_arr).any():
			print("Nan detectect", np_arr)

		group.create_dataset("gray_image", data=np_arr, compression="gzip", compression_opts=9)

		if debug:
			print(msg_topic)
			print("height: {}, width: {}".format(msg.height, msg.width))
			print("encoding: {}".format(msg.encoding))
			print("step in bytes: {}".format(msg.step))
			print(np_arr.shape)
			# reshape from 1d to 2d to plot the image properly
			np_arr = np.reshape(np_arr, (height, width))
			print(np_arr.shape)
	# im = Image.fromarray(np_arr, "I;16")
	# im.show()
	# img = Image.fromarray(np_arr)
	# plt.imshow(img)
	# plt.pause(0.01)

	if msg_topic == t_pico_depth:
		np_arr = np.fromstring(msg.data, np.uint16)
		group.create_dataset("depth_image", data=np_arr, compression="gzip", compression_opts=9)

		if debug:
			print(msg_topic)
			print("height: {}, width: {}".format(msg.height, msg.width))
			print("encoding: {}".format(msg.encoding))
			print("step in bytes: {}".format(msg.step))
	# # reshape from 1d to 2d to plot the image properly
	# np_arr = np.reshape(np_arr, (1, height, width))
	# print(np_arr.shape)
	# img_depth = Image.fromarray(np_arr, "F;32")
	# img = Image.fromarray(np_arr)
	# plt.imshow(img_depth)
	# plt.pause(0.01)

	if msg_topic == t_pico_pointcloud:
		if debug:
			print(msg_topic)
			print("height", msg.height)
			print("width", msg.width)
			print("point_step", msg.point_step)
			print("point_row", msg.row_step)
			print("is dense", msg.is_dense)

		''' 
		https://www.programcreek.com/python/example/99841/sensor_msgs.msg.PointCloud2
			Converts a rospy PointCloud2 message to a numpy recordarray 

			Assumes all fields 32 bit floats, and there is no padding.
			'''
		dtype_list = [(f.name, np.float32) for f in msg.fields]
		cloud_arr = np.fromstring(msg.data, dtype_list)
		group.create_dataset("pointcloud", data=cloud_arr, compression="gzip", compression_opts=9)

	if msg_topic == t_atracsys_head:
		if debug:
			print(msg_topic)
			print("atracsys msg:", msg.transform.translation, msg.transform.rotation)

		atracsys_head = np.zeros(7)
		tf = []

		tf.append(msg.transform.translation.x)
		tf.append(msg.transform.translation.y)
		tf.append(msg.transform.translation.z)
		tf.append(msg.transform.rotation.x)
		tf.append(msg.transform.rotation.y)
		tf.append(msg.transform.rotation.z)
		tf.append(msg.transform.rotation.w)

		rob6_tf = np.asarray(tf)
		group.create_dataset("trk_head_tf", data=rob6_tf, compression="gzip", compression_opts=9)

	if msg_topic == t_atracsys_coil:
		if debug:
			print(msg_topic)
			print("atracsys msg:", msg.transform.translation, msg.transform.rotation)

		atracsys_coil = np.zeros(7)
		tf = []

		tf.append(msg.transform.translation.x)
		tf.append(msg.transform.translation.y)
		tf.append(msg.transform.translation.z)
		tf.append(msg.transform.rotation.x)
		tf.append(msg.transform.rotation.y)
		tf.append(msg.transform.rotation.z)
		tf.append(msg.transform.rotation.w)

		rob6_tf = np.asarray(tf)
		group.create_dataset("trk_coil_tf", data=atracsys_coil, compression="gzip", compression_opts=9)
def save_topics_clean(group, msg_topic, msg):
	bridge = CvBridge()
	if msg_topic == t_rob6_joints:
		joints = np.asarray(msg.position)
		group.create_dataset("joints", data=joints, compression="gzip", compression_opts=9)

	if msg_topic == t_pico_gray:
		gray = bridge.imgmsg_to_cv2(msg)
		group.create_dataset("gray", data=gray, compression="gzip", compression_opts=9)

	if msg_topic == t_pico_depth:
		depth = bridge.imgmsg_to_cv2(msg)
		# cv2.imshow("Depth", depth)
		# cv2.waitKey(3)
		group.create_dataset("depth", data=depth, compression="gzip", compression_opts=9)

	if msg_topic == t_rob6_tf:
		t = np.zeros((1, 3))
		t[0, 0] = msg.transform.translation.x
		t[0, 1] = msg.transform.translation.y
		t[0, 2] = msg.transform.translation.z
		group.create_dataset("translation", data=t, compression="gzip", compression_opts=9)
		q = np.zeros((1, 4))
		q[0, 0] = msg.transform.rotation.w
		q[0, 1] = msg.transform.rotation.x
		q[0, 2] = msg.transform.rotation.y
		q[0, 3] = msg.transform.rotation.z
		a = pq.Quaternion(q[0])
		q = a.normalised
		# convert quaternion to numpy array
		r_list = np.asarray([q.w, q.x, q.y, q.z])
		# print(q)
		# print(r_list)
		group.create_dataset("rotation", data=r_list, compression="gzip", compression_opts=9)
def ros_to_h5():
	sample = 0


	array = [1, 2, 3]
	out_array = np.asarray(array)

	width = 224
	height = 171

	gray_image = np.zeros((height, width))
	depth_image = np.zeros((height, width))
	pointcloud = np.zeros((height, width))

	trk_head_tf = np.zeros(7)
	trk_coil_tf = np.zeros(7)

	hf_train = h5.File('kub_train.h5', 'w')
	hf_val = h5.File('kub_val.h5', 'w')

	# estimate number of messages in bag file
	msg_count = 0
	try:
		with rosbag.Bag(filename, 'r') as bag:
			# save counter to calculate the 80/20 split
			msg_count = bag.get_message_count()
	except rosbag.bag.ROSBagException as err:
		print(err.value)

	# load the first 80 % into train set
	split_ratio = 0.8
	train_size = math.ceil(msg_count / 7.0 * split_ratio)
	# load the last 20 % into val set
	val_size = math.ceil(msg_count / 7.0 - train_size)
	print("Train: {} Val: {}".format(train_size, val_size))

	samples = 1
	count = 1
	upper_bound = train_size + val_size
	topics_list = [t_rob6_joints, t_rob6_tf, t_pico_gray, t_pico_depth, t_pico_pointcloud]
	topic_count = len(topics_list)
	try:
		with rosbag.Bag(filename, 'r') as bag:
			for msg_topic, msg, t in bag.read_messages(
					topics=topics_list):

				# control logic
				# count until all 7 messages have been saved
				# increase sample counter + 1

				group_name = "/sample-" + str(sample)

				if sample % 500 == 0 and count == topic_count:
					print("Sample progress: [{:d} {:.1f} // {:3.2f} %]".format(sample, msg_count / topic_count, (
							float(sample) / (msg_count / topic_count)) * 100))

				# print("{} < {} <= {}".format(train_size, sample, upper_bound))
				if not sample % 5 == 0:
					# if not sample % math.ceil(100/val_size) == 0:
					#     print("Split ratio:", math.ceil((msg_count/7.0)/val_size))
					if group_name in hf_train and count == topic_count:
						# print(sample, sample % 5, "train")
						count = 0
						sample = sample + 1
						group_name = "/sample-" + str(sample)
					elif group_name not in hf_train and count < topic_count:
						g_train = hf_train.create_group(group_name)
					save_topics(g_train, msg_topic, msg)
				else:
					if group_name in hf_val and count == topic_count:
						# print(sample, sample % 5, "val")
						count = 0
						sample = sample + 1
						group_name = "/sample-" + str(sample)
					elif group_name not in hf_val and count < topic_count:
						g_validation = hf_val.create_group(group_name)
					save_topics(g_validation, msg_topic, msg)

				# print("Train", len(hf_train.items()), list(hf_train.items()))
				# print("Val", len(hf_val.items()), list(hf_val.items()))

				# print("Train:", len(hf_train.items()))
				# print("Val: ", len(hf_val.items()))
				# print("\n")

				count = count + 1

			# sample = sample + 1
			print("total msgs: {}".format(sample))
			hf_train.close()
			hf_val.close()
	except rosbag.bag.ROSBagException as err:
		print(err.value)
def read_h5():
	path = "/home/kub/Downloads/masterthesis-master/kub_train.h5"
	# path = "./dataset_val_v4.h5"
	# d = DatasetFromHdf5.DatasetFromHdf5(path)
	# print(len(d))
	hf = h5.File(path, "r")
	# for i in hf.items():
	# 	print(i, i.keys())
	# print("key", k)

	total_items = len(hf.items())
	random_samples = []
	total_images = 25

	for i in range(total_images):
		random_samples.append(randrange(total_items))

	images = []
	count = 0
	for i in hf.items():
		print(i[0])
		group = hf.get(i[0])
		gray = np.array(group.get("gray"))
		img = PIL_Image.fromarray(gray)
		img = img.rotate(90)
		plt.imshow(img)
		plt.pause(0.01)
		plt.savefig("img_" + str(i[0]) + ".svg", format="svg", dpi=600)
		if count > 25:
			break
		else:
			count += 1

	for j, i in enumerate(random_samples):
		group = hf.get("sample" + str(i))
		#images.append(np.array(group.get("gray")))
		#gray = np.array(group.get("gray"))
		img = PIL_Image.fromarray(gray)
		img = img.rotate(90)
		fig = plt.imshow(img)
		plt.axis("off")
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		plt.pause(0.001)
		plt.savefig("img_gray_" + str(i) + ".svg", format="svg", dpi=600, bbox_inches="tight", pad_inches=0)
# print(images[0])
# width = images[0].shape[0]
# height = images[0].shape[1]
# canvas = np.zeros((width*total_images, height*total_images))

# print(canvas.shape)
# canvas[0:height][0:width] = images[0]
# for i in canvas:
# 	print(i)
# print(canvas[0].shape)
# img = PIL_Image.fromarray(canvas)
# plt.imshow(img)
# plt.pause(10)


# print(group.keys())
# group = hf.get(k)
# g_items = list(group.items())
# print("group items: ", g_items)
# gray = np.array(group.get("gray"))
# depth = np.array(group.get("depth"))*400

# np_arr = np.reshape(dataset, (171, 224))
# print(np_arr.shape)
# im = Image.fromarray(np_arr, "I;16")
# im_gray = PIL_Image.fromarray(gray)
# im_depth = PIL_Image.fromarray(depth)

# im_gray.show()
# im_depth.show()
# im_gray.save("img_gray.png")
# im_depth.save("img_depth.png")

# img = PIL_Image.fromarray(depth)
# img = img.rotate(90)
# plt.imshow(img)
# plt.pause(10)
# plt.savefig("img_depth.svg", format="svg", dpi=600)


def create_dataset(bag_filename, mode):
	# load rosbag file
	# estimate number of messages in bag file
	msg_count = 0
	print("Loading file {:s}".format(bag_filename))
	try:
		with rosbag.Bag(bag_filename, 'r') as bag:
			# save counter to calculate the 80/20 split
			msg_count = bag.get_message_count()
			topics = bag.get_type_and_topic_info().topics
	except rosbag.bag.ROSBagException as err:
		print(err.value)

	print("total number of messages", msg_count, msg_count / len(topics))
	print("topics", topics)
	print(len(topics))
	size = msg_count / len(topics)

	translations = np.zeros((size, 3))
	rotations = np.zeros((size, 4))

	start = time.time()
	sample = 0
	count = 1
	print("Loading data from rosbag file")
	try:
		with rosbag.Bag(filename, 'r') as bag:
			for msg_topic, msg, t in bag.read_messages(
					topics=[t_rob6_joints, t_rob6_tf, t_pico_gray, t_pico_depth, t_pico_pointcloud, t_atracsys_head,
					        t_atracsys_coil]):

				if msg_topic == t_rob6_tf:
					# print("sample", sample, msg_topic)
					# print(msg)
					# add translations
					t = np.zeros((1, 3))
					t[0, 0] = msg.transform.translation.x
					t[0, 1] = msg.transform.translation.y
					t[0, 2] = msg.transform.translation.z
					# translations = np.append(translations, t)
					translations[sample] = t
					# np.insert(translations,sample,t)

					# get the quaternion from msg
					# print(msg.transform.rotation)
					q = np.zeros((1, 4))
					q[0, 0] = msg.transform.rotation.w
					q[0, 1] = msg.transform.rotation.x
					q[0, 2] = msg.transform.rotation.y
					q[0, 3] = msg.transform.rotation.z
					# print(q)
					# print(q[0])
					a = pq.Quaternion(q[0])
					q = a.normalised
					# rotations[sample, :] = q
					rotations[sample, 0] = q[0]
					rotations[sample, 1] = q[1]
					rotations[sample, 2] = q[2]
					rotations[sample, 3] = q[3]
					# print("a", a)
					# print("q", q)
					# print()
					# print("rotations", rotations)
					sample = sample + 1

				count = count + 1
				duration = time.time() - start
				print("Duration {:s}".format(time.strftime("%H:%M:%S", time.gmtime(duration))))
	except rosbag.bag.ROSBagException as err:
		print(err.value)
	print("Finished Loading")
	print("\n" * 2)



	# calc all translations
	# uncomment from here
	# calculate L2 norm matrix of translations
	'''''''''''
	dist_trans = np.zeros((size, size))
	print(dist_trans)

	print("Start calculating Translation Distance")

	for i, item_i in enumerate(translations):
		for j, item_j in enumerate(translations):
			l2 = math.sqrt(
				(item_i[0] - item_j[0]) ** 2 + (item_i[1] - item_j[1]) ** 2 + (item_i[2] - item_j[2]) ** 2)

			dist_trans[i, j] = l2

	sio.savemat(mode + "_dist_trans_v3.mat", {"dist_trans": dist_trans})
	

	'''''
	#uncomment the block above if you want to creat a dist_trans.mat file
	dist_quat = np.zeros((size, size))
	#print(rotations)

	print("Start calculating rotational distance")

	total_sec = len(rotations) / 50 * 47
	start = time.time()
	duration = 0
	# calc the rotational distance between quaternions
	for i, item_i in enumerate(rotations):
		for j, item_j in enumerate(rotations):
			t = transformation.Transformation(config=None)
			# print(item_i)
			# print(item_j)
			# print(np.asarray(item_i))
			rot_dist = t.quaternion_distance(item_i, item_j)[0]
			dist_quat[i, j] = rot_dist
			print("Dist quad : ", dist_quat[i,j])

		if i % 10 == 0:
			duration = time.time() - start
			remaining_time = total_sec - duration
			# print(
			# 	i, j, rot_dist, "{:2.2%}".format(float(i) / j),
			# 	time.strftime("%H:%M:%S", time.gmtime(remaining_time)))


	sio.savemat(mode + "_dist_quat_v3.mat", {"dist_quat": dist_quat})
def create_relative_dataset(file_dir, mode):
	# read h5 file
	hf = h5.File(file_dir, "r")
	total_items = int(len(hf.items())*0.01)
	print("Total Items", total_items)

	translations = np.zeros((total_items, 3))
	rotations = np.zeros((total_items, 4))
	sample = 0
	count = 0
	for i in hf.items():
		for j in range(0,total_items):
			print(i)
			group = hf.get(i[0])
			# print(group)
			translation = np.array(group.get("translation"))
			translation = translation[0]
			#print(translation[0], len(translations[sample]))
			# print(translation)
			count = count +1
			# get translation vector
			t = np.zeros((1, 3))
			t[0, 0] = translation[0]
			t[0, 1] = translation[1]
			t[0, 2] = translation[2]
			print(len(t), count)
			translations[sample] = t

			# get the quaternion rotation from rotation
			rotation = np.array(group.get("rotation"))
			q = np.zeros((1, 4))
			q[0, 0] = rotation[0]
			q[0, 1] = rotation[1]
			q[0, 2] = rotation[2]
			q[0, 3] = rotation[3]
			a = pq.Quaternion(q[0])
			q = a.normalised
			# print(q)
			rotations[sample, 0] = q[0]
			rotations[sample, 1] = q[1]
			rotations[sample, 2] = q[2]
			rotations[sample, 3] = q[3]
			sample = sample + 1
		break



	# calc all translations
	# calculate L2 norm matrix of translations
	dist_trans = np.zeros((total_items, total_items))

	print("Start calculating Translation Distance")


	for i, item_i in enumerate(translations):
		for j, item_j in enumerate(translations):
			l2 = math.sqrt(
				(item_i[0] - item_j[0]) ** 2 + (item_i[1] - item_j[1]) ** 2 + (item_i[2] - item_j[2]) ** 2)
			print("Translation Progress {:5.1%} Distance Rot: {:5.2f}".format((float(len(translations))*len(translations)), l2))
			dist_trans[i, j] = l2
	# print(dist_trans)
	file_name = mode + "_" + file_dir.split(".")[0] + "_" + "translation"+".mat"
	sio.savemat(file_name, {"dist_trans": dist_trans})
	## uncomment the block above if you want to creat a dist_trans.mat file.
	dist_quat = np.zeros((total_items, total_items))
	print(rotations)

	print("Translation Finished")


	print("Start calculating rotational distance")

	total_sec = 10000
	speed = 100
	start = time.time()
	duration = 0
	# calc the rotational distance between quaternions
	for i, item_i in enumerate(rotations):
		for j, item_j in enumerate(rotations):
			speed_start = time.time()
			t = transformation.Transformation(config=None)
			#print(item_i)
			#print(item_j)
			#print(np.asarray(item_i))
			print(item_i, item_j)
			rot_dist = t.quaternion_distance(item_i, item_j)[0]
			dist_quat[i, j] = rot_dist
			# speed = time.time() - speed_start
			# total_sec = math.ceil(len(rotations)/speed)
		if i % 10 == 0:
			duration = time.time() - start
			# remaining_time = total_sec - duration
			# print(
			# 	i, j, rot_dist, "{:2.2%}".format(float(i) / j),
			# 	time.strftime("%H:%M:%S", time.gmtime(remaining_time)))
			print(
				"Combinations {:5d} {:5d} Rot-Dist: {:3.2f} Progress {:2.1%} Duration {:s}".format(i, j, rot_dist,
				                                                                                    float(i) / j,
				                                                                                    time.strftime(
					                                                                                    "%H:%M:%S",
					                                                                                    time.gmtime(
						                                                                                    duration))))
	# print(dist_quat)
	file_name = mode + "_" + file_dir.split(".")[0] + "_" + "rotation"+".mat"
	#sio.savemat(file_name, {"dist_quat": dist_quat})

	print("Rotation Finished")
def test_numpy():
	total = np.zeros((4, 3))
	l = np.array([1, 2, 3])
	print(total)
	print(l)
	# total = np.insert(total, 0, l)
	total[0] = l
	total[1] = l
	print(total[0])
	# total = np.insert(total, 1, l)
	print(total)
def plot_heatmap():
	t = sio.loadmat("train_dist_quat_v2.mat")
	data = t["dist_quat"]
	print(np.shape(data))
	print(data)
	im = plt.imshow(data, cmap='hot', interpolation='nearest')
	plt.colorbar(im)
	plt.title("Translational Distance Heatmap")
	plt.show()
def plot_surface():
	t = sio.loadmat("train_result_t5_r5_v7.mat")
	data = t["result"]
	print(np.shape(data))
	print(data)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	x = y = np.arange(0, np.shape(data)[0])
	X, Y = np.meshgrid(x, y)
	Z = data.reshape(X.shape)

	ax.scatter(X, Y, Z)

	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	plt.show()
def test_poseloss():
	pred = np.array([[1.2308e+01, -8.3917e+00, 1.4943e+01, -2.4882e-05, 5.8604e-01,
	                  -4.7962e-01, 6.6435e-01]])
	label = np.array([[1.2294e+01, -8.3870e+00, 1.4929e+01, -3.3931e-03, 5.8330e-01,
	                   -4.8346e-01, 6.6693e-01]])

	# eucledian distance translation error
	print(pred[0][0:3])
	print(pred[0][0:7])
	x = np.sum((pred[0][0:3] - label[0][0:3]) ** 2)
	print(x)
	print(math.sqrt(x))

	print(pred[0][3:7])
	a = pq.Quaternion(pred[0][3:7])
	print(a)
	b = pq.Quaternion(label[0][3:7])
	print(a.norm)
	print("a normalized", a / a.norm)
	print(a.normalised)
	print(b.normalised)

	t = transformation.Transformation(config=None)
	print(t.quaternion_distance(a, b)[1])
	print(t.quaternion_distance(a.normalised, b.normalised)[1])
def test_quaternions():
	a = [-0.00435228, -0.00435228, -0.00435228, -0.00435228]
	b = [-0.01940142, -0.01940142, -0.01940142, -0.01940142]
	x = pq.Quaternion(a)
	y = pq.Quaternion(b)
	print(x)
	print(x.normalised)
	print(y)
	print(y.normalised)
	t = transformation.Transformation(config=None)
	print(t.quaternion_distance(x.normalised, y.normalised))
def filter_translation(dir_file, t=40, binary=True):
	file = sio.loadmat(dir_file)
	trans = file["dist_trans"]
	trans_filtered = np.zeros(np.shape(trans))

	for x in range(np.shape(trans)[0]):
		for y in range(np.shape(trans)[1]):
			# print(trans[x, y])
			if trans[x, y] <= t:
				if binary:
					trans_filtered[x, y] = 1
				else:
					trans_filtered[x, y] = trans[x, y] / t

	# im = plt.imshow(trans_filtered, cmap='hot', interpolation='nearest')
	# plt.colorbar(im)
	# plt.title("Translational Distance Filtered")
	# plt.show()
	return trans_filtered
def filter_rotation(dir_file, r=40, binary=True):
	file = sio.loadmat(dir_file)
	rot = file["dist_quat"]
	rot_filtered = np.zeros(np.shape(rot))

	for x in range(np.shape(rot)[0]):
		for y in range(np.shape(rot)[1]):
			# print(rot[x, y])
			if rot[x, y] <= r:
				if binary:
					rot_filtered[x, y] = 1
				else:
					rot_filtered[x, y] = rot[x, y] / r

	# im = plt.imshow(rot_filtered, cmap='hot', interpolation='nearest')
	# plt.colorbar(im)
	# plt.title("rotational Distance Filtered")
	# plt.show()

	return rot_filtered
def merge_distance_matrices(filter_trans, filter_rot, dir_trans, dir_rot, mode):

	#train_result mat dosyasi olusturuluyor

	print("Loaded file {:s}".format(dir_trans))
	print("Loaded file {:s}".format(dir_rot))
	# filter translations under 4 mm
	a = filter_translation(dir_file=dir_trans, t=filter_trans)
	# filter rotations under 2 degrees
	b = filter_rotation(dir_file=dir_rot, r=filter_rot)

	res = np.multiply(a, b)
	total_combinations = np.size(res)
	total_filtered = np.count_nonzero(res)
	print("a:",a, "b:", b,"total combinations:", total_combinations, "total filtered:",total_filtered)
	#print("Filtered {:2.2%}".format(float(total_filtered) / total_combinations))
	file_name = mode+"_"+"result_t{:d}_r{:d}_v7".format(filter_trans, filter_rot)
	sio.savemat(file_name+".mat", {"result": res})

	im = plt.imshow(res, cmap='plasma', interpolation='nearest')
	plt.colorbar(im)
	plt.title("Rotation and Translation Filtered Combined")
	plt.xlabel("Samples")
	plt.ylabel("Samples")
	# plt.show()
	plt.savefig(file_name+".svg".format(filter_trans, filter_rot), dpi=1200, format="svg")
	print("Saved svg file")
def analyse_data():
	d = sio.loadmat("train_result_t5_r5_v7.mat")
	data = d["result"]
	print(np.size(data))
	print("Non Zeros Entries: ", np.count_nonzero(data))
	print("Allowed Neighbors {:3.2f} %".format(float(np.count_nonzero(data)) / np.size(data) * 100))
def load_bagfile():
	# load rosbag file
	# estimate number of messages in bag file
	msg_count = 0
	try:
		with rosbag.Bag(rosbag_dir, 'r') as bag:
			# save counter to calculate the 80/20 split
			msg_count = bag.get_message_count()
			topics = bag.get_type_and_topic_info().topics
	except rosbag.bag.ROSBagException as err:
		print(err.value)

	print("total number of messages", msg_count, msg_count / len(topics))
	print("topics", topics)
	print(len(topics))
	size = msg_count / len(topics)

	entries = {}
	entries["translation"] = None
	entries["rotation"] = None
	entries["gray"] = None
	entries["depth"] = None
	entries["joints"] = None

	bag_data = []
	bag_data.append(entries)
	# translations = np.zeros((size, 3))
	# rotations = np.zeros((size, 4))
	path = "dataset_clean_v3_" + str(datetime.datetime.now()) + ".h5"
	ds = h5.File(path, "w")

	sample = 0
	count = 0
	progress = 0
	group_name = "/sample" + str(sample)
	group = ds.create_group(group_name)
	print("Loading data from rosbag file")

	topics_list = [t_rob6_joints, t_rob6_tf, t_pico_gray, t_pico_depth]
	# message_count = 0
	# try:
	# 	with rosbag.Bag(filename, 'r') as bag:
	# 		message_count = bag.get_message_count()
	# except rosbag.bag.ROSBagException as err:
	# 	print(err.value)

	start = time.time()
	# speed = 10
	# total_sec = len(message_count) / 1 * speed

	bridge = CvBridge()
	try:
		with rosbag.Bag(rosbag_dir, 'r') as bag:
			for msg_topic, msg, t in bag.read_messages(topics=topics_list):
				# time_speed = time.time()
				if count == len(topics_list):
					sample = sample + 1
					group_name = "/sample" + str(sample)
					entries = {}
					entries["translation"] = None
					entries["rotation"] = None
					entries["gray"] = None
					entries["depth"] = None
					entries["joints"] = None
					bag_data.append(entries)
					group = ds.create_group(group_name)
					count = 0
					duration = time.time() - start
					print("Progress: {:2.2%}".format(float(sample) / msg_count),
					      time.strftime("%H:%M:%S", time.gmtime(duration)))

				if msg_topic == t_rob6_joints:
					joints = np.asarray(msg.position)
					bag_data[sample]["joints"] = joints
					group.create_dataset("joints", data=joints, compression="gzip", compression_opts=9)

				if msg_topic == t_pico_gray:
					gray = bridge.imgmsg_to_cv2(msg)
					# cv2.imshow("Gray", gray*100)
					# cv2.waitKey(3)
					bag_data[sample]["gray"] = gray
					group.create_dataset("gray", data=gray, compression="gzip", compression_opts=9)

				if msg_topic == t_pico_depth:
					depth = bridge.imgmsg_to_cv2(msg)
					# cv2.imshow("Depth", depth)
					# cv2.waitKey(3)
					bag_data[sample]["depth"] = depth
					group.create_dataset("depth", data=depth, compression="gzip", compression_opts=9)

				if msg_topic == t_rob6_tf:
					t = np.zeros((1, 3))
					t[0, 0] = msg.transform.translation.x
					t[0, 1] = msg.transform.translation.y
					t[0, 2] = msg.transform.translation.z
					bag_data[sample]["translation"] = t
					group.create_dataset("translation", data=t, compression="gzip", compression_opts=9)
					# print("Translation", t)
					# print(bag_data[sample]["translation"])
					q = np.zeros((1, 4))
					q[0, 0] = msg.transform.rotation.w
					q[0, 1] = msg.transform.rotation.x
					q[0, 2] = msg.transform.rotation.y
					q[0, 3] = msg.transform.rotation.z
					a = pq.Quaternion(q[0])
					q = a.normalised
					bag_data[sample]["rotation"] = q
					# convert quaternion to numpy array
					r_list = np.asarray([q.w, q.x, q.y, q.z])
					# print(q)
					# print(r_list)
					group.create_dataset("rotation", data=r_list, compression="gzip", compression_opts=9)

				count = count + 1


	except rosbag.bag.ROSBagException as err:
		print(err.value)

	print("Finished H5 Dataset")
	print("Items in ds", ds.items())
	ds.close()

	print("Finished Loading")
	print("Loaded {} samples".format(len(bag_data)))
	return bag_data
def generate_dataset_train_val(ros_bag_dir, name_train, name_val, percent_train_use=0.8, clean=False):
	width = 224
	height = 171

	hf_train = h5.File(name_train, "w")
	hf_val = h5.File(name_val, "w")

	# estimate number of messages in bag file
	print("open bag file", ros_bag_dir)
	msg_count = 0
	try:
		with rosbag.Bag(ros_bag_dir, 'r') as bag:
			# save counter to calculate the 80/20 split
			msg_count = bag.get_message_count()
	except rosbag.bag.ROSBagException as err:
		print(err.value)

	# load the first 80 % into train set
	split_ratio = percent_train_use
	# train_size = math.ceil(msg_count / 7.0 * split_ratio)
	# load the last 20 % into val set
	# val_size = math.ceil(msg_count / 7.0 - train_size)
	# print("Train: {} Val: {}".format(train_size, val_size))

	sample = 0
	count = 1
	topics_list = [t_rob6_joints, t_rob6_tf, t_pico_gray, t_pico_depth, t_pico_pointcloud]
	topic_count = len(topics_list)

	print("sampling h5 file")
	try:
		with rosbag.Bag(ros_bag_dir, 'r') as bag:
			for msg_topic, msg, t in bag.read_messages(
					topics=topics_list):

				# control logic
				# count until all 7 messages have been saved
				# increase sample counter + 1

				group_name = "/sample-" + str(sample)

				if sample % 500 == 0 and count == topic_count:
					print("Sample progress: [{:d} {:.1f} // {:2.1%} ]".format(sample, msg_count / topic_count, (
							float(sample) / (msg_count / topic_count))))

				mod = int((1-split_ratio)**(-1))
				if not sample % mod == 0 or clean == True:
					if group_name in hf_train and count == topic_count:
						# print("Train {:d} {:d}".format(sample, sample % mod))
						count = 0
						sample = sample + 1
						group_name = "/sample-" + str(sample)
					elif group_name not in hf_train and count < topic_count:
						g_train = hf_train.create_group(group_name)
					# save_topics(g_train, msg_topic, msg)
					save_topics_clean(g_train, msg_topic, msg)
				else:
					if group_name in hf_val and count == topic_count:
						# print("Val {:d} {:d}".format(sample, sample % mod))
						count = 0
						sample = sample + 1
						group_name = "/sample-" + str(sample)
					elif group_name not in hf_val and count < topic_count:
						g_validation = hf_val.create_group(group_name)
					# save_topics(g_validation, msg_topic, msg)
					save_topics_clean(g_validation, msg_topic, msg)

				count = count + 1

			print("total msgs: {}".format(sample))
			hf_train.close()
			hf_val.close()
	except rosbag.bag.ROSBagException as err:
		print(err.value)

def generate_dataset():
	# load result.mat file
	f_neighbor = sio.loadmat("train_result_t5_r5_v7.mat")
	neighbors_mat = f_neighbor["result"]

	# load rosbag file
	data = load_bagfile()
	print(data[0])
	# loop through matrix
	print(len(data))
	print(len(data[0]))
	# print(data[-1])

	print(len(neighbors_mat))
	# print(len(neighbors_mat[0]))
	start = time.time()
	max = len(data)
	path = "test_train_clean.h5"
	dataset_h5_train = h5.File(path, "w")
	sample = 0
	for x, dim0 in enumerate(neighbors_mat):
		duration = time.time() - start
		sec = duration % 60
		min = (duration / 60) % 60
		hour = (duration / 3600)
		if x % 2 == 0:
			print("Progress {:3.2f} % | {:5d} | {:02d}:{:02d}:{:02d}".format(float(x) / max * 100, x,
			                                                                 int(hour), int(min), int(sec)))
		for y, dim1 in enumerate(dim0):
			if dim1 != 0:
				group_name = "/sample-" + str(sample)
				group = dataset_h5_train.create_group(group_name)
				group.create_dataset("gray_template", data=data[x]["gray"], compression="gzip",
				                     compression_opts=9)
				group.create_dataset("gray_live", data=data[y]["gray"], compression="gzip",
				                     compression_opts=9)
				group.create_dataset("depth_template", data=data[x]["depth"], compression="gzip",
				                     compression_opts=9)
				group.create_dataset("depth_live", data=data[y]["depth"], compression="gzip",
				                     compression_opts=9)
				t_delta = data[x]["translation"] - data[y]["translation"]
				t_delta = np.asarray(t_delta[0])
				group.create_dataset("t_delta", data=t_delta, compression="gzip", compression_opts=9)
				r_delta = data[x]["rotation"] - data[y]["rotation"]
				# convert quaternion to numpy array
				r_list = []
				r_list.append(r_delta[0])
				r_list.append(r_delta[1])
				r_list.append(r_delta[2])
				r_list.append(r_delta[3])
				r_list = np.asarray(r_list)
				group.create_dataset("r_delta", data=r_list, compression="gzip", compression_opts=9)
				sample += 1

	dataset_h5_train.close()
	print("Dataset successfully created {}".format(path))
def parser_new():
	filename = '/home/kub/Downloads/rosbag_converter/step_and_shoot.bag'
	t_pico_gray = "/royale_camera_driver/depth_image"
	width = 224
	height = 171

	try:
		with rosbag.Bag(filename, 'r') as bag:
			for msg_topic, msg, t in bag.read_messages(topics=[t_pico_gray]):
				# print(msg.encoding)
				bridge = CvBridge()
				cv_image = bridge.imgmsg_to_cv2(msg)
				cv_image = cv_image * 1
				print(cv_image.shape)
				print(type(cv_image))

				cv2.imshow("Image window", cv_image)
				cv2.waitKey(3)

		# np_arr = np.fromstring(msg.data, np.uint16)
		# np_arr = np.reshape(np_arr, (height, width))
		# print(np_arr.shape)
		# im = Image.fromarray(np_arr, "I;16")
		# # im.show()
		# img = Image.fromarray(np_arr)
		# plt.imshow(img)
		# plt.pause(0.01)
	except rosbag.bag.ROSBagException as err:
		print(err.value)
def test_convert_mat_to_list(result_mat, target_csv):
    # file_mat = "/home/Brendes/rosbag_converter/result_t4_r2.mat"
    d = io.loadmat(result_mat)
    print("loaded {:s}".format(result_mat))
    mat = d["result"]
    # print(mat)
    # print(np.shape(mat))
    # print(np.size(mat))

    # faster way of creating a lut
    # reduce mat table
    mat_new = []
    line = []
    print("iterating through mat file")
    for x, item_x in enumerate(mat):
        for y, item in enumerate(mat[x]):
            if item == 1:
                # print(x, y, item)
                line.append(y)
        mat_new.append(line)
        line = []
    # print(mat_new)
    # print(np.shape(mat_new))
    # print(np.size(mat_new))

    # save mat_new to csv file
    # with open("./datasets/mat_new.csv", "w", newline="") as f:
    print("saving data into {:s}".format(target_csv))
    with open(target_csv, "w") as f:
        writer = csv.writer(f)
        writer.writerows(mat_new)
    f.close()
    print("saved {:s}".format(target_csv))


if __name__ == "__main__":

	#parser_new()
	#ros_to_h5()
	#read_h5()
	#plot_heatmap()

	#load_bagfile()
	#generate_dataset()
	create_dataset('step_and_shoot.bag', mode ="train" ) #dist_quat mtri olusturuluyor

	#merge_distance_matrices()




	## read rosbag file into h5
	#ros_to_h5()
	#generate_dataset_train_val(ros_bag_dir=rosbag_dir, name_train="kub_train.h5", name_val="kub_val.h5", clean=True)

	# calc relative neighbor matrices
	#val = "kub_val.h5"
	#create_relative_dataset(file_dir=val, mode="val")
	#train = "kub_train.h5"
	#create_relative_dataset(file_dir=train, mode="train")

	## merge the two val_dist_quat and val_dist_trans matrices
	# merge_distance_matrices(filter_trans=5, filter_rot=5, dir_trans="val_test_val_translation.mat", dir_rot="val_test_val_rotation.mat", mode="val")
	#merge_distance_matrices(filter_trans=5, filter_rot=5, dir_trans="val_kub_val_translation.mat", dir_rot="val_kub_val_rotation.mat" , mode="train")

	## convert the result mat lists into a csv file
	#test_convert_mat_to_list(result_mat="val_result_t5_r5_v7.mat", target_csv="val_result_t5_r5_v7.csv")
	#test_convert_mat_to_list(result_mat="train_result_t5_r5_v7.mat", target_csv="train_result_t5_r5_v7.csv")

	#merge_distance_matrices(filter_trans=40, filter_rot=40, dir_trans= "train_dist_trans_v3.mat", dir_rot= "train_dist_quat_v3.mat", mode = "train")
