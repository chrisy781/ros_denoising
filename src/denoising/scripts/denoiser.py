#!/usr/bin/env python3
import rospy
import pcl
import sensor_msgs.point_cloud2 as pc2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from scipy.spatial import cKDTree
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

def ground_plane_removal(points):
    # 1) For estimating the ground plane: only regard lowest z value points
    z_threshold = np.percentile(points[:, 2], 35)  
    mask = (points[:, 2] < z_threshold) 
    plane_points = points[mask]

    x = plane_points[:, 0]
    y = plane_points[:, 1]
    z = plane_points[:, 2]

    # Reshape x and y to column vectors
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    X = np.column_stack((x, y))

    # 2) Fit a RANSAC model to the data: for GP estimation
    ransac = RANSACRegressor(LinearRegression(), random_state=0)
    ransac.fit(X, z)

    # Extract the coefficients
    a, b, d = ransac.estimator_.coef_[0], ransac.estimator_.coef_[1], ransac.estimator_.intercept_
    
    # Remove plane
    x = points[:,0]
    y = points[:,1]
    z_adjustment = np.mean(plane_points[:,2]) # plane was too high
    z = -(a * x + b * y) + z_adjustment # was +d
    intensity = np.full_like(x, fill_value=255)
    plane = np.column_stack((x, y, z, intensity))

    # Visualize plane - not applicable
    num_points = 1000
    x_p = np.random.uniform(low=-30, high=30, size=num_points)
    y_p = np.random.uniform(low=-30, high=30, size=num_points)
    z_p = -(a * x_p + b * y_p) + z_adjustment
    i_p = np.full_like(x_p, fill_value=255)
    vis_plane = np.column_stack((x_p, y_p, z_p, i_p))

    # filter out ground plane
    margin = 0.25  # 10 cm margin
    mask_z = (np.abs(points[:, 2] - z) < margin) 
    plane_less_cloud = points[~mask_z]

    return vis_plane, plane_less_cloud

def count_neighbors_inliers(kdtree, query_point, search_radius):
    # Perform a radius (ball) search
    neighbors_indices = kdtree.query_ball_point(query_point, search_radius)
    # Count the total number of neighbors
    num_neighbors = len(neighbors_indices)
    return num_neighbors

def denoiser(points): # AGDOR rebuild
    # --- set params --- 
    intensity_thresh = 9 
    alpha = 0.1 
    angular_res = 0.1 
    min_neighbors = 5 
    
    # --- stage 1 --- intensity filter (should be based labelled data of different dataset - IS NOT ATM!)
    mask = points[:,3] < intensity_thresh 
    p_o = points[~mask] 
    p_i_low = points[mask]

    # --- stage 2 --- adaptive group density outlier removal
    p_dist = np.linalg.norm(p_i_low, axis=1)
    SR = alpha*angular_res*p_dist # apply a dynamic search radius
    print("Median search radius = ", np.median(SR))
    p_core = []
    p_outlier = []
    kdtree = cKDTree(p_i_low[:,0:3])

    for index, (query_point, search_radius) in enumerate(zip(p_i_low, SR)):
        num_neighbors = count_neighbors_inliers(kdtree, query_point[0:3], search_radius)
        # print(num_neighbors)
        if min_neighbors <= num_neighbors: 
            p_core.append(list(query_point))
        elif min_neighbors > num_neighbors:
            p_outlier.append(list(query_point))

    p_outlier = np.array(p_outlier)
    p_core = np.array(p_core)

    if np.size(p_core)!=0:
        p_stacked = np.vstack((p_o, p_core))
        unique_points, indices = np.unique(p_stacked, axis=0, return_index=True)
        p_o = p_stacked[np.sort(indices)]
        return p_o
    else:
        print("WARNING: the SR did not return any points!")
        return p_o 
    

def callback(raw_cloud):
    rospy.loginfo("Processing the point cloud")

    # Convert the raw PointCloud2 to a NumPy array
    cloud_data = pc2.read_points(raw_cloud, field_names=("x", "y", "z", "intensity"), skip_nans=True)
    cloud_array = np.array(list(cloud_data), dtype=np.float32)

    # --- applying the denoising method ---
    _, planeless_cloud = ground_plane_removal(cloud_array)
    filtered_cloud_array = denoiser(planeless_cloud)

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