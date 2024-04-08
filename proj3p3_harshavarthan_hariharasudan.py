#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray
from geometry_msgs.msg import Twist
import numpy as np
import heapq
import cv2
import math
import time
import matplotlib.pyplot as plt

start_time = time.time()
# ------------------------------------------------------------
# ----------------------DEFINING THE MAP----------------------
# ------------------------------------------------------------
# empty map
map_image = np.zeros((200, 600, 3), dtype=np.uint8)
# radius = int(input("enter the radius of the robot: ")) #radius of the robot
clearance = int(input("enter the obstacle clearance (range from 1 to 10): "))  # clearance of the obstacles
RPM1 = int(input("enter the left wheel velocity:"))
RPM2 = int(input("enter the right wheel velocity:"))
radius = 22
dt = 0.1

# clearance = 10

class VelocityPublisher(Node):
    def __init__(self):
        super().__init__('velocity_publisher')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        time.sleep(2)  # Give some time for the publisher to set up

    def publish_velocity(self, linear_velocity, angular_velocity):
        msg = Twist()
        msg.linear.x = linear_velocity
        msg.angular.z = angular_velocity
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % msg)

def RoundToFifteen(x, base=15):
    return base * round(x / base)


# obsatcles initialization
for i in range(0, 601):  # 6000 width
    for j in range(0, 201):  # 2000 height

        # The fist rectangle
        if (150 - clearance - radius <= i <= 175 + clearance + radius and 0 <= j <= 100 + clearance + radius):
            cv2.circle(map_image, (int(i), int(j)), 2, (0, 255, 0), -1)

        # second rectangle
        elif (250 - clearance - radius <= i <= 275 + clearance + radius and 100 - clearance - radius <= j <= 200):
            cv2.circle(map_image, (int(i), int(j)), 2, (0, 255, 0), -1)

        # circle obstacle
        elif ((i - 400) ** 2 + (j - 80) ** 2 <= (60 + radius + clearance) ** 2):
            cv2.circle(map_image, (int(i), int(j)), 2, (0, 255, 0), -1)


        # boundries
        elif (j <= radius + clearance or j >= 200 - radius - clearance):
            cv2.circle(map_image, (int(i), int(j)), 2, (0, 255, 0), -1)
        elif (i <= radius + clearance or i >= 600 - radius - clearance):
            cv2.circle(map_image, (int(i), int(j)), 2, (0, 255, 0), -1)

        # everything else is white
        else:
            cv2.circle(map_image, (int(i), int(j)), 2, (255, 255, 255), -1)

# displaying the map after alloting the relevant space for radius and clearance
cv2.imshow('nap', map_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

height, width, _ = map_image.shape  # Getting the shape of the map


# function to check if a node is free(not an obstacle or out of bounds)
def is_free(x, y, free_space):
    # Invert y-coordinate
    inverted_y = height - y - 1
    if 0 <= x < width and 0 <= inverted_y < height:
        # Check if the cell is free (precomputed)
        return free_space[inverted_y, x]
    return False


# Function to find the heuristic values
def heuristic(node, goal):
    return math.sqrt((node[0] - goal[0]) ** 2 + (node[1] - goal[1]) ** 2)


# Function to check if the goal is reached
def is_goal_reached(current_node, goal_node, distance_tolerance=5):
    distance = heuristic(current_node, goal_node)
    # checking if the current node is with in a distance of 1.5 from the goal node
    return distance <= distance_tolerance


explored_nodes = []


# --------------------------------------------------------------
# -------------------DEFINING THE ACTION SET-------------------
# --------------------------------------------------------------

def ActionMove(curr_node, orientation_facing, RPM_L, RPM_R, d=1.0, L=28.7, step_size=1.0):
    t = 0
    r = 3.3  # Radius of the wheel
    Xn, Yn = curr_node[0], curr_node[1]
    vl = RPM_L * (2 * np.pi)/60
    vr = RPM_R * (2 * np.pi)/60
    Thetan = np.deg2rad(orientation_facing)  # Converting to radians directly

    while t < 1:
        t += dt
        dX = 0.5 * r * (vl + vr) * math.cos(Thetan) * dt
        dY = 0.5 * r * (vl + vr) * math.sin(Thetan) * dt
        dTheta = (r / L) * (vr - vl) * dt
        
        # linear_velocity = 0.5 * r * (RPM_L + RPM_R) * math.cos(Thetan) * dt
        # angular_velocity = (r / L) * (RPM_R - RPM_L) * dt
        Xn += dX
        Yn += dY
        Thetan += dTheta
    # print(vel)

    final_orientation = np.rad2deg(Thetan) % 360
    final_orientation = RoundToFifteen(final_orientation)  # Ensure orientation is rounded as per requirement
    return (Xn, Yn, final_orientation), _,_,_


# -----------------------------------------------------------
# -----------------------A* ALGORITHM------------------------
# -----------------------------------------------------------

def a_star(map_image, start, goal, step_size):
    queue = [(heuristic(start, goal), start)]
    visited = set()  # This set will store visited nodes to avoid revisiting them
    parents = {start: None}
    costs = {start: 0}
    free_space = np.all(map_image == [255, 255, 255], axis=2)  # Precompute free space
    child_node = []
    parent_node = []
    k = 0
    action_king = {start: None}
    actions = [[0, RPM1], [RPM1, 0], [RPM1, RPM1], [0, RPM2], [RPM2, 0], [RPM2, RPM2], [RPM1, RPM2], [RPM2, RPM1]]
    while queue:
        _, current_node = heapq.heappop(queue)
        current_node_rounded = (round(current_node[0]), round(current_node[1]), current_node[2])

        visited.add(current_node_rounded)  # Add the current node to the visited set
        child = []
        if is_goal_reached(current_node, goal):
            return backtrack(parents, start, current_node, action_king), costs[
                current_node], explored_nodes, parent_node, child_node

        for action in actions:
            next_node, linear_velocity, angular_velocity,veal = ActionMove(current_node[:2], current_node[2], action[0], action[1])
            next_node_rounded = (round(next_node[0]), round(next_node[1]), next_node[2])

            print("Current node:", current_node)
            print("Next node:", next_node[:2], "Step size (heuristic):", heuristic(current_node, next_node))

            # Skip this node if it's already visited or not free
            if next_node_rounded in visited or not is_free(int(next_node[0]), int(next_node[1]), free_space):
                continue

            new_cost = costs[current_node] + heuristic(current_node, next_node)
            # print("individual cost")
            # print(heuristic(current_node,next_node))

            if next_node not in costs or new_cost < costs[next_node]:
                costs[next_node] = new_cost
                priority = new_cost + heuristic(next_node, goal)*2
                print("Next node:", next_node[:2], "Step size (heuristic):", priority)

                # print("cost")
                # print(priority)
                heapq.heappush(queue, (priority, next_node))
                parents[next_node] = current_node
                action_king[next_node] = (action, linear_velocity,angular_velocity)
                explored_nodes.append(next_node)
                child.append(next_node[:2])
                k = k+1

        if child == []:
            continue
        else:
            child_node.append(child)
            parent_node.append(current_node[:2])
    return None, None, explored_nodes, parent_node, child_node  # Goal not reached


def backtrack(parents, start, goal, action_king):
    path = [goal]
    actions_taken = []
    velocities_taken = []  # To store velocities
    current = goal
    while current != start:
        if current in parents:
            actions_taken.insert(0, action_king[current][0])  # Get the action
            velocities_taken.insert(0, action_king[current][1:])  # Get the velocities
            current = parents[current]
            path.insert(0, current)
        else:
            break
    return path, actions_taken, velocities_taken


# For visualization and main execution logic, you can use the same as provided in your original code, ensuring it calls 'visualize' correctly with the generated path.
def visualize(canvas_BGR, path, parent, child, explored_nodes):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('output_harshavarthan_hariharasudan.mp4', fourcc, 20.0, (width, height))

    c1 = 0  # counter for explored nodes
    c2 = 0  # counter for node path

    # Drawing the start in green
    cv2.circle(canvas_BGR, (int(start[0]), int(height - start[1] - 1)), 5, (0, 255, 0), -1)
    # drawing the goal in blue
    cv2.circle(canvas_BGR, (int(goal[0]), int(height - goal[1] - 1)), 5, (255, 0, 0), -1)
    i = 0

    # Drawing the vectors of the explored nodes
    for sp in parent:
        nursery = child[i]
        for baby in nursery:
            cv2.arrowedLine(canvas_BGR, (int(sp[0]), int(height - sp[1] - 1)),
                            (int(baby[0]), int(height - baby[1] - 1)), [200, 160, 40], 1, tipLength=0.2)
            # cv2.circle(canvas_BGR,(int(baby[0]),int(height-baby[1])), 1, (0,255,255),-1)
        i = i + 1
        if i % 1000 == 0:
            video.write(canvas_BGR)

    cv2.circle(canvas_BGR, (int(start[0]), int(height - start[1] - 1)), 5, (0, 255, 0), -1)
    # drawing the goal in blue
    cv2.circle(canvas_BGR, (int(goal[0]), int(height - goal[1] - 1)), 5, (255, 0, 0), -1)

    # drawing the path in red
    for node in path[0]:
        cv2.circle(canvas_BGR, (int(node[0]), int(height - node[1] - 1)), 5, (0, 0, 255), -1)
        cv2.circle(canvas_BGR, (int(node[0]), int(height - node[1] - 1)), 1, (0, 0, 0), -1)
        video.write(canvas_BGR)

    for i in range(35):
        video.write(canvas_BGR)

    # release the video writer
    print("Total Time Taken : ", time.time() - start_time, "seconds")
    video.release()


def main(path):
    rclpy.init()
    velocity_publisher = VelocityPublisher()
    try:
        for velocity in path:
            oritent = velocity
            linear_vel, angular_vel = oritent[0] , oritent[1]
            velocity_publisher.publish_velocity(linear_vel, angular_vel)
            time.sleep(1)
        velocity_publisher.publish_velocity(0.0, 0.0)  # Stop the robot
    except KeyboardInterrupt:
        pass
    finally:
        velocity_publisher.destroy_node()
        rclpy.shutdown()

# if __name__ == '__main__':
#     # main(velocities)

if __name__ == "__main__":
    free_space = np.all(map_image == [255, 255, 255], axis=2)

    valid_start = False
    valid_goal = False

    # Keep prompting until valid start and goal are provided
    while not valid_start or not valid_goal:
        # Get start coordinates and orientation
        start_x, start_y, start_theta = map(int, input(
            "Enter start coordinates (x,y) and orientation theta (degrees), split by a comma(',')(example input:50,100,0): ").split(
            ','))

        # Ensuring orientation is a multiple of 15 degrees
        if start_theta % 30 != 0:
            print("Orientation must be in increments of 30 degrees.")
            continue

        # Get goal coordinates
        goal_x, goal_y = map(int, input(
            "Enter goal coordinates (x,y), split by a comma(',')(example input:575,100): ").split(','))
        # Check if the start and goal nodes are in an obstacle or outside the image bounds
        if not is_free(start_x, start_y, free_space):
            print("The start position is invalid or within an obstacle.")
        else:
            valid_start = True

        if not is_free(goal_x, goal_y, free_space):
            print("The goal position is invalid or within an obstacle.")
        else:
            valid_goal = True

    # If both start and goal are valid, proceed with path planning
    start = (start_x, start_y, start_theta)
    goal = (goal_x, goal_y)
    step_size = 1  # step size
    V = []
    path, cost, explored_nodes, parent, child = a_star(map_image, start, goal, step_size)
    print("A* search is over.")
    if path:
        print(f"Node path: {path[0]}")  # print the path
        print("The actions required to reach the end:")
        print(path[1])
        print(f"Cost: {cost}")  # print the cost
        print(len(path))

        print("velocities :", path[2])  # printing the velocities required

        visualize(map_image, path, parent, child, explored_nodes)  # initialize the visualization
        r = 3.3
        L = 28.7
        vel = []
        for i in path[1]:
            print("-----------------------------------------------")
            print(i)
            lin = i[0]
            ang = i[1]
            linear_velocity = (r/2)*(2*np.pi*(lin/60)+2*np.pi*(ang/60))/100 # Linear velocity in some units per time
            angular_velocity = (r/L)*(2*np.pi*(ang/60)-2*np.pi*(lin/60))  # Angular velocity in radians per time
            print("respective linear", linear_velocity, "   respective angular   ", angular_velocity)
            vel.append((linear_velocity,angular_velocity))
        main(vel)
    else:
        print("Cannot find the path.")

