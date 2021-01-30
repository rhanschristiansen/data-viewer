# viewer
Video Lidar Viewer

This viewer allows the user to view video and lidar data in an interactive GUI based system

## Install dependencies
1. python3 venv
   ```bash
   sudo apt-get install -y python3-venv
   ```
   be sure to create/activate the venv
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
1. `wxPython4`: Manually install pip wheel for ubuntu 18.04 (replace with 16.04 if needed)
   ```bash
   pip install -U \
    -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04 \
    wxPython
   ```
1. Install requirements.txt: 
   ```bash
   python3 -m pip install -r requirements.txt
   ```
## Running the Viewer
### Download data
TODO: image and video data not in cloud storage yet, update when available.  
### Running in PyCharm
1. Set Python interpreter in Settings to `venv/bin/python3`  
1. Navigate to [src/viewer.py](src/viewer.py)  
1. Click on Play button  
![image](doc/play-button.png)
   
### Running from terminal
1. Navigate to root folder

1. Set the `PYTHONPATH` root dir as `$root_dir/src`
    ```bash
    cd src/
    export PYTHONPATH=$PWD
    ```
1. Run the viewer
    ```bash
    python3 viewer.py
    ```

## Generating Bounding Boxes for Kitti and Santa Clara datasets
See [src/util/preprocess_santaclara.py](src/util/preprocess_santaclara.py) and [src/util/preprocess_kitti.py](src/util/preprocess_kitti.py)