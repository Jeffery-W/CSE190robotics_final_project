#!/usr/bin/env python

#robot.py implementation goes here
import rospy
import numpy as np
from read_config import read_config
from std_msgs.msg import Bool
from cse_190_final_project.msg import PolicyList
from qlearning import qlearning


class Robot():

    def __init__(self):
        self.config = read_config()
        self.map_height = self.config['map_size'][0]
        self.map_width = self.config['map_size'][1]
        rospy.init_node('robot')
        rospy.sleep(1)
        self.policy_publisher = rospy.Publisher(
                '/results/policy_list',
                PolicyList,
                queue_size=self.config['max_runs']*100
        ) 
        self.sim_complete = rospy.Publisher(
                '/map_node/sim_complete',
                Bool,
                queue_size=10
        )
        # run mdp and qlearning
        rospy.sleep(2)
        self._run_simulation()
        rospy.sleep(10)
        self.sim_complete.publish(Bool(True))
        rospy.sleep(10)
        rospy.signal_shutdown("Simulation completed.")
    
    def _run_simulation(self):
        policies = qlearning(self.config)
        for policy in policies:
            self.policy_publisher.publish(PolicyList(policy))

if __name__ == '__main__':
    robot = Robot()

