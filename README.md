## TensorRT Download

### [Link](https://developer.nvidia.com/tensorrt-download)

- Select according to Windows and CUDA version

download and unzip
<br>
<br>
Open terminal and go to python folder
```
Example
1. pip install python\tensorrt-8.6.1.6-cp37-none-win_amd64.whl
2. pip install uff\uff-0.6.9-py2.py3-none-any.whl
3. pip install graphsurgeon\graphsurgeon-0.4.5-py2.py3-none-any.whl
4. pip install onnx_graphsurgeon\onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl

warning : When running once, select the version that matches the installed Python version.
cp : Python version
cp36 : Python 3.6
cp37 : Python.3.7
cp38 : Python.3.8
cp39 : Python.3.9
cp310 : Python.310
cp311 : Python.3.11

* TensorRT Version : 8.6.1.6
```


<br>


## Download Sample Data

Install the tool dependencies via python3 -m pip install -r requirements.txt.
Invoke downloader.py to download the data with a command like the one below if download.yml is present in the sample directory (example).
```
downloader.py -d /path/to/data/dir -f /path/to/download.yml
```
The data directory i.e. /path/to/data/dir is a centralized directory to store data of all samples. So you can use same one for all samples. It can be provided by either -d /path/to/data/dir or the environment variable $TRT_DATA_DIR, where the -d has higher priority.


Remember to use -d or $TRT_DATA_DIR when running sample scripts that rely on downloaded data. Scripts will abort if no downloaded data is found in data directory. ($TRT_DATA_DIR will be much simplier.) An error will be thrown if the data is not properly setup.
The download.yml file is owned by the sample which describes the sample name, the path, URL and checksum of the data files that are required by the sample.

#### Notes for sample developers

To use the downloaded data files, integrate the code segment like below into the sample code, and obtain the path to the data file by passing the path as specified in the associated download.yml file of the sample.
```
TRT_DATA_DIR = None

def getFilePath(path):
    global TRT_DATA_DIR
    if not TRT_DATA_DIR:
        parser = argparse.ArgumentParser(description="Convert YOLOv3 to ONNX model")
        parser.add_argument('-d', '--data', help="Specify the data directory where it is saved in. $TRT_DATA_DIR will be overwritten by this argument.")
        args, _ = parser.parse_known_args()
        TRT_DATA_DIR = os.environ.get('TRT_DATA_DIR', None) if args.data is None else args.data
    if TRT_DATA_DIR is None:
        raise ValueError("Data directory must be specified by either `-d $DATA` or environment variable $TRT_DATA_DIR.")

    fullpath = os.path.join(TRT_DATA_DIR, path)
    if not os.path.exists(fullpath):
        raise ValueError("Data file %s doesn't exist!" % fullpath)

    return fullpath
```

The helper function getFilePath in downloader.py can also be used to obtain the full path to the downloaded data files. It only works when the sample doesn't have any other command line argument.

```
from downloader import getFilePath

cfg_file_path = getFilePath('samples/python/yolov3_onnx/yolov3.cfg')
```

![image](https://github.com/JeHeeYu/Yolo-to-trt-Build-Guide/assets/87363461/9cb266ae-ad65-443b-916b-ed6efb350cae)

<br>

## Prerequisites

![image](https://github.com/JeHeeYu/Yolo-to-trt-Build-Guide/assets/87363461/40487ade-9741-421c-83ec-1ae93dd08c11)

After downloading yolov3.weight, yolov3.cfg, dog.png, place the files in [TensorRT]/samples/python/yolov3_onnx/

### [Link](https://pjreddie.com/darknet/yolo/)

When changing file name and settings, change in download.yml
```
download.yml

sample: yolov3_onnx
files:
  - path: samples/python/yolov3_onnx/yolov3.cfg
    url: https://raw.githubusercontent.com/pjreddie/darknet/f86901f6177dfc6116360a13cc06ab680e0c86b0/cfg/yolov3.cfg
    checksum: b969a43a848bbf26901643b833cfb96c

  - path: samples/python/yolov3_onnx/yolov3.weights
    url: https://pjreddie.com/media/files/yolov3.weights
    # mirror: https://master.dl.sourceforge.net/project/darknet-yolo.mirror/darknet_yolo_v3_optimal/yolov3.weights
    checksum: c84e5b99d0e52cd466ae710cadf6d84c

  - path: samples/python/yolov3_onnx/dog.jpg
    url: https://github.com/pjreddie/darknet/raw/f86901f6177dfc6116360a13cc06ab680e0c86b0/data/dog.jpg
    checksum: 0efe2b8fa0609cf67d33ad9ed8112e66

```

<br>

Install the dependencies for Python

```
pip3 install -r requirements.txt
```

## Running the sample

The data directory needs to be specified (either via -d /path/to/data or environment varaiable TRT_DATA_DIR) when running these scripts. An error will be thrown if not. Taking TRT_DATA_DIR approach in following example.

```
python3 yolov3_to_onnx.py -d [TensorRT Folder]

Example
python3 yolov3_to_onnx.py -d C:\TensorRT-8.6.1.6
```

When running the above command for the first time, the output should look similar to the following:

```
[...]
%106_convolutional = Conv[auto_pad = u'SAME_LOWER', dilations = [1, 1], kernel_shape = [1, 1], strides = [1, 1]]
(%105_convolutional_lrelu, %106_convolutional_conv_weights, %106_convolutional_conv_bias)
return %082_convolutional, %094_convolutional,%106_convolutional
}
```

<br>

Build a TensorRT engine from the generated ONNX file and run inference on a sample image

```
python3 onnx_to_tensorrt.py -d [TensorRT Folder]

Example
python3 yolov3_to_onnx.py -d C:\TensorRT-8.6.1.6
```

When running the above command for the first time, the output should look similar to the following:

```
Building an engine from file yolov3.onnx, this may take a while...
Running inference on image dog.jpg...
Saved image with bounding boxes of detected objects to dog_bboxes.jpg.
```

Verify that the sample ran successfully. If the sample runs successfully you should see output similar to the following:

```
Loading ONNX file from path yolov3.onnx...
Beginning ONNX file parsing
Completed parsing of ONNX file
Building an engine from file yolov3.onnx; this may take a while...
Completed creating Engine
Running inference on image dog.jpg...
[[135.14841333 219.59879284 184.30209195 324.0265199 ]
  [ 98.30805074 135.72613533 499.71263299 299.25579652]
  [478.00605802 81.25702449 210.57787895 86.91502688]] [0.99854713 0.99880403 0.93829258] [16 1 7]
Saved image with bounding boxes of detected objects to dog_bboxes.png.
```

<br>

## Result

The yolov3.trt file and dog_bboxes.png are created in the folder.

![image](https://github.com/JeHeeYu/Yolo-to-trt-Build-Guide/assets/87363461/cd01b55e-4ebc-400d-89ee-ff20a83adaff)


<br>

### [Source](https://github.com/NVIDIA/TensorRT/tree/release/8.6/samples/python/yolov3_onnx) : NVIDIA/TensorRT Github
