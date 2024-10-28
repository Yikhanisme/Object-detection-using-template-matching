# Object detection using template matching
## Introduction
Template Matching is one of the simplest and most effective techniques for object detection, particularly when the target object’s appearance and orientation are relatively consistent.

Template matching is a technique used to detect and locate an object (template) within a larger image by comparing small image patches. It works by sliding the template image across the target image (the input) and computing a similarity measure for each position.

This method compares the pixel values between the template and the input image patch, generating a score that indicates the degree of match at each location. The areas with the highest scores are considered the best match.

## Installation
### Prerequisites
Ensure you have Python 3.7 or higher installed, along with pip to manage dependencies.

### Install Dependencies
1. Clone the repository:
```bash
git clone https://github.com/Bao-NQ06/Object-detection-using-template-matching.git
cd Object-detection-using-template-matching
```
2. Enviroment setup: Recommends creating a virtual environment and installing dependencies via pip. Use the following commands to set up your environment:
We also recommend using pytorch>=2.0 and cuda>=11.8, though lower versions of PyTorch and CUDA are supported.

* On Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```
* On MacOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install Dependencies:
Make sure to install all dependencies listed in requirements.txt:
```bash
pip install -r requirements.txt
```
4. How to use:
* If you already have a template image and a image you want to detect. Run the following code to run the local program:
```bash
streamlit run app.py
```
The program will run in your local host.
If you want to close the local webbapp. Press Ctrl + c in terminal.

* If you want to crop template from your image to find all similar object in your image:
```bash
python app_v2.py
```
- Use your mouse to crop the object
- Press c to crop and save the object to 'cropped_image.jpg' or Press r to reset image
- Press q to quit cropping and processing image
- The result will be save to result folder as 'result_2.png'

5. Deactivate the Virtual Environment: When you are done, deactivate the virtual environment by running:
```bash
deactivate
```
