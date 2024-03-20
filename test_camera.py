import cv2  
  
# 海康威视摄像头的RTSP URL  
rtsp_url = "rtsp://[username]:[password]@[camera_ip_address]:[port]/Streaming/Channels/[channel_id]"  
  
# 替换URL中的占位符  
# 例如: rtsp://admin:123456@192.168.1.64:554/Streaming/Channels/101  
rtsp_url = rtsp_url.replace('[username]', 'admin')       # 替换为你的用户名  
rtsp_url = rtsp_url.replace('[password]', '12A34a56')     # 替换为你的密码  
rtsp_url = rtsp_url.replace('[camera_ip_address]', '192.168.1.130')  # 替换为摄像头的IP地址  
rtsp_url = rtsp_url.replace('[port]', '554')              # 替换为RTSP端口，通常是554  
rtsp_url = rtsp_url.replace('[channel_id]', '101')        # 替换为通道ID，根据摄像头配置而定  
  
# 创建VideoCapture对象  
cap = cv2.VideoCapture(rtsp_url)  
  
# 检查是否成功打开视频流  
if not cap.isOpened():  
    print("Error: Could not open video stream or file")  
    exit()  
  
# 无限循环显示视频帧  
while True:  
    # 逐帧捕获  
    ret, frame = cap.read()  
      
    # 如果正确读取帧，ret为True  
    if not ret:  
        print("Unable to receive frame")  
        break  
  
    # 显示结果帧  
    cv2.imshow('Frame', frame)  
      
    # 等待1ms，然后检查退出键（'q'）是否被按下  
    if cv2.waitKey(1) & 0xFF == ord('q'):  
        break  
  
# 完成后，释放cap对象  
cap.release()  
cv2.destroyAllWindows()
