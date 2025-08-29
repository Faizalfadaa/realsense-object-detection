#!/usr/bin/env python3

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image           
from std_msgs.msg import Int64, String      
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloNode(Node):
    def __init__(self):
        super().__init__('YoloNode')

        # ==============================================
        # PARAMETER DECLARATION
        # ==============================================

        # Declare parameters for topic names
        self.declare_parameter('color_image_raw_topic','/camera/camera/color/image_raw') # msg type sensor_msgs/msg/Image -> Menerima gambar dari Depth Camera
        self.declare_parameter('detection_model', '/home/alfadha/ros2_ws/src/realsense-object-detection/yolo_detection/yolo_detection/yolov8n.pt')
        self.declare_parameter('threshold_val', 0.45)
        self.declare_parameter('result_image_topic', 'money_detect/image')
        self.declare_parameter('count_money_topic', 'money_detect/count')
        self.declare_parameter('total_money_topic', 'money_detect/total')

        # ==============================================
        # PARAMETER RETRIEVAL
        # ==============================================

        # Get Parameter values
        color_image_raw_topic = self.get_parameter('color_image_raw_topic').get_parameter_value().string_value

        result_image_topic = self.get_parameter('result_image_topic').get_parameter_value().string_value
        count_money_topic = self.get_parameter('count_money_topic').get_parameter_value().string_value
        total_money_topic = self.get_parameter('total_money_topic').get_parameter_value().string_value

        detection_model = self.get_parameter('detection_model').get_parameter_value().string_value
        threshold_val = self.get_parameter('threshold_val').get_parameter_value().double_value
        self.conf = float(threshold_val)

        self.detection_model = YOLO(detection_model)
        self.bridge = CvBridge()

        # ==============================================
        # PUBLISHERS
        # ==============================================

        # Create Publisher
        self.result_image_pub = self.create_publisher(
            Image,
            result_image_topic,
            10
        )
        self.count_money_pub = self.create_publisher(
            Int64,
            count_money_topic,
            10
        ) 
        self.total_money_pub = self.create_publisher(
            Int64,
            total_money_topic,
            10
        )

        # ==============================================
        # SUBSCRIBERS
        # ==============================================

        # Create Subscriber
        self.color_image_raw_sub = self.create_subscription(
            Image,
            color_image_raw_topic,
            self.color_image_raw_callback,
            10
        )

        # ==================================================
        # SUBSCRIBER CALLBACK
        # ==================================================
    
    def color_image_raw_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg,'bgr8')
        result = self.model(frame, conf= self.conf, verbose= False)[0]

        money_value = {
            '2000': 2000,
            '5000': 5000,
            '10000': 10000,
            '50000': 50000,
            '100000': 100000
        }
        total = 0
        n = 0

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            cls_id = int(box.cls[0].item())
            label = self.model.names[cls_id]

            nominal = money_value.get(label, 0)  
            total += nominal
            n += 1

            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(img, f'{label} ({nominal:,})', (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(img, f'Count: {n}   Total: Rp {total:,}', (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,200,50), 2)

        self.result_image_pub.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))
        self.count_money_pub.publish(Int64(n))
        self.total_money_pub.publish(Int64(total))            

        # ==================================================
        # TIMER CALLBACK
        # ==================================================

def main(args=None):
    rclpy.init(args=args)
    node = NodeTemplate()
   
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
