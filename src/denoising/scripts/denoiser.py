#!/usr/bin/env python3
import rospy
import pcl
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField


def denoiser(cloud_array):
    # Extract intensities
    intensities = cloud_array[:, 3]

    # Calculate the median intensity
    median_intensity = np.median(intensities)

    # Filter out points with intensities below the median
    mask = intensities >= median_intensity
    filtered_cloud_array = cloud_array[mask]

    return filtered_cloud_array


def callback(raw_cloud):
    rospy.loginfo("Processing the point cloud")

    # Convert the raw PointCloud2 to a NumPy array
    cloud_data = pc2.read_points(raw_cloud, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    cloud_array = np.array(list(cloud_data), dtype=np.float32)

    # --- applying the denoising method ---
    filtered_cloud_array = denoiser(cloud_array)

    # Setting up the PointCloud2 message which includes the filtered pointcloud data
    filtered_cloud_msg = PointCloud2()
    filtered_cloud_msg.header = raw_cloud.header
    filtered_cloud_msg.header.frame_id = 'map' 
    filtered_cloud_msg.height = 1
    filtered_cloud_msg.width = filtered_cloud_array.shape[0]

    # Define the fields
    filtered_cloud_msg.fields = [
        pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
        pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
        pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
        pc2.PointField(name="intensity", offset=12, datatype=pc2.PointField.FLOAT32, count=1)]

    # Set the point step
    filtered_cloud_msg.point_step = 16  # 4 fields * 4 bytes per field

    # Convert the filtered point cloud array back to bytes and assign it to the data field
    filtered_cloud_msg.data = filtered_cloud_array.tobytes()

    # Publish the denoised point cloud
    pub.publish(filtered_cloud_msg)

if __name__ == '__main__':
    rospy.init_node('denoising_node', anonymous=True)
    sub = rospy.Subscriber('point_cloud_topic', PointCloud2, callback) # reading the raw point cloud topic for data to be proccessed
    pub = rospy.Publisher('denoised_cloud', PointCloud2, queue_size=10) # this is the topic to which the processed point gets publsihed
    rate = rospy.Rate(1)  # Adjust the rate as needed

    rospy.spin()