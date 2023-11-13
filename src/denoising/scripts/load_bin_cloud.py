#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2, PointField
import pcl
import numpy as np
import struct

def open_cloud():
    file_path = "/home/chris/Documents/labelled test data (WADS)/velodyne/039498.bin"

    with open(file_path, "rb") as file:
        data = file.read()  # Read the binary data from the file

    # Assuming the .bin file contains 4 floats per point (x, y, z, intensity)
    # Convert binary data into numpy array
    cloud_data = np.frombuffer(data, dtype=np.float32)
    cloud_data = cloud_data.reshape(-1, 4)  # Reshape data into points (4 values per point)

    # Convert the numpy array to PCL PointCloud
    cloud = pcl.PointCloud_PointXYZI()
    cloud.from_array(cloud_data) 
    print(cloud_data.shape)
    return cloud_data, cloud

def publish_point_cloud():
    rospy.init_node('point_cloud_publisher', anonymous=True)
    pub = rospy.Publisher('point_cloud_topic', PointCloud2, queue_size=10)

    # Read data from the .bin file and convert it to PointCloud format using PCL
    cloud_data, cloud = open_cloud()
    
    # Create a PointCloud2 message manually
    cloud_msg = PointCloud2()
    cloud_msg.header.stamp = rospy.Time.now()
    cloud_msg.header.frame_id = 'map' 
    # cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = cloud_data.shape[0]  # Total number of points
    cloud_msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1)
    ]
    cloud_msg.is_bigendian = False
    cloud_msg.point_step = 16  # Size of a single point in bytes (4 floats * 4 bytes)
    cloud_msg.row_step = cloud_msg.point_step * cloud_data.shape[0]
    cloud_msg.is_dense = True

    # Pack the point cloud data into a byte array
    cloud_msg.data = struct.pack('f' * len(cloud_data.ravel()), *cloud_data.ravel())

    # Get and log the first point for debugging purposed
    first_point = cloud[0]  
    rospy.loginfo("First point of the cloud: {}".format(first_point))
   

    rate = rospy.Rate(1)  # Adjust the rate as needed
    while not rospy.is_shutdown():
        # Publish the point cloud data
        pub.publish(cloud_msg)
        rate.sleep()


if __name__ == '__main__':
    try:
        publish_point_cloud()
    except rospy.ROSInterruptException:
        pass

