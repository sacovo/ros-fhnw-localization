#!/usr/bin/env python
import os
import csv

from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
import click


# Create a typestore and get the string class.
typestore = get_typestore(Stores.LATEST)


def to_miliseconds(time, sim=False):
    return time.sec * 1000 + time.nanosec / 1e6


def extract_imu(msg):
    return [to_miliseconds(msg.header.stamp), msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    
def extract_odom(msg):
    return [to_miliseconds(msg.header.stamp), msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z, 
            msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w,
            msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z, msg.twist.twist.angular.x,
            msg.twist.twist.angular.y, msg.twist.twist.angular.z]
    
@click.command()
@click.option("--imu_topic", default="/imu", help="The IMU topic to extract.")
@click.option("--odom_topic", default="/odomimu", help="The odometry topic to extract.")
@click.argument("bag_paths", nargs=-1, type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path(dir_okay=True, file_okay=False))
@click.option("--skip-existing/--no-skip-existing", default=True)
def main(bag_paths, output_dir, imu_topic, odom_topic, skip_existing=True):

    for bag_path in bag_paths:

        if bag_path.endswith("/"):
            bag_path = bag_path[:-1]

        output_dir_ = os.path.join(output_dir, os.path.basename(bag_path))

        if (not os.path.isdir(bag_path)) or (skip_existing and os.path.exists(output_dir_)):
            print(f"Skipping {bag_path}, {output_dir_}")
            continue
        extract_bag(bag_path, output_dir_, imu_topic, odom_topic)
        print(f"Extracted {imu_topic} and {odom_topic} from {bag_path} to {output_dir_}")

def extract_bag(bag_path, output_dir, imu_topic, odom_topic):
    # Open the bag file for reading
    os.makedirs(output_dir, exist_ok=True)

    files = {topic: open(f'{output_dir}/{topic}.csv', "w") for topic in ['imu', 'odom']}
    writers = {topic: csv.writer(file) for topic, file in files.items()}
    
    writers["imu"].writerow(["timestamp", "angular_velocity_x", "angular_velocity_y", "angular_velocity_z",
                           "linear_acceleration_x", "linear_acceleration_y", "linear_acceleration_z",
                           "orientation_x", "orientation_y", "orientation_z", "orientation_w"])
    writers["odom"].writerow(["timestamp", "position_x", "position_y", "position_z", "orientation_x", "orientation_y",
                            "orientation_z", "orientation_w", "linear_velocity_x", "linear_velocity_y", "linear_velocity_z",
                            "angular_velocity_x", "angular_velocity_y", "angular_velocity_z"])

    with Reader(bag_path) as bag:
        
        # Iterate through the bag file and extract messages with the specified topic
        for connection, ts, data in bag.messages():
            # Write the timestamp and message values to the CSV file
            if connection.topic == imu_topic:
                msg = typestore.deserialize_cdr(data, connection.msgtype)
                writers['imu'].writerow(extract_imu(msg))
            elif connection.topic == odom_topic:
                msg = typestore.deserialize_cdr(data, connection.msgtype)
                writers["odom"].writerow(extract_odom(msg))
        
    for file in files.values():
        file.close()
        

if __name__ == "__main__":
    main()
