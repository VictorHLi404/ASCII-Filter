This is an open-source CLI tool used to apply an ASCII art filter onto any source video, including a live camera output. It leverages the [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) deep learning model and OpenCV's computer vision library in Python to convert video content to ASCII art based on brightness and estimated depth of objects in frame.

**The program is relatively resource-heavy due to the cost of running the local deep learning model on your machine. If it runs slowly, please give it time to load.**

To run the program, do the following:

1. If on a device that contains an NVIDIA GPU that supports CUDA, follow the instructions to [download the CUDA driver](https://developer.nvidia.com/cuda-downloads) for your device. CUDA acceleration will greatly improve the ease of use and quality of output of the application.
2.  Have a working version of Python 3.13+.
3.  Clone the repository to your local computer. 
4. Run `pip install requirements.txt` in either a local or virtual environment.
5. Run `python -m main` in your console. The CLI prompt will activate and run from there.

