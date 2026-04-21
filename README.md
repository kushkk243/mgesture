
## How to clone and run the script inside docker everytime.


```bash
cd /root/m_ws/src

git clone https://github.com/kushkk243/mgesture.git

cd ..

colcon build --packages-select mgesture

source install/setup.bash

ros2 run mgesture gesture_publisher
```
