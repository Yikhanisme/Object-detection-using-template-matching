import streamlit as st 
import cv2
import numpy as np
import os
from PIL import Image
from MTM import matchTemplates


def preprocess_image(image_path):
    if image_path is None:
        raise ValueError(f"Failed to load image at {image_path}.")
    return cv2.cvtColor(image_path, cv2.COLOR_BGR2GRAY)

def non_max_suppression(boxes, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    boxes = np.array(boxes, dtype="float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    scores = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:-1]])
        yy1 = np.maximum(y1[i], y1[idxs[:-1]])
        xx2 = np.minimum(x2[i], x2[idxs[:-1]])
        yy2 = np.minimum(y2[i], y2[idxs[:-1]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:-1]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

def multi_scale_template_matching(image, template, min_scale=0.05, max_scale=1.5, threshold=0.4):
    """Perform multi-scale template matching and apply NMS to filter results."""
    found_boxes = []
    template_height, template_width = template.shape[:2]

    for scale in np.linspace(min_scale, max_scale, 50)[::-1]:
        resized = cv2.resize(template, (int(template_width * scale), int(template_height * scale)))

        if resized.shape[0] > image.shape[0] or resized.shape[1] > image.shape[1]:
            continue

        result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)

        if max_val >= threshold:
            top_left = max_loc
            bottom_right = (top_left[0] + resized.shape[1], top_left[1] + resized.shape[0])
            found_boxes.append([top_left[0], top_left[1], bottom_right[0], bottom_right[1], max_val])

    nms_boxes = non_max_suppression(np.array(found_boxes), overlapThresh=0.3)

    return nms_boxes


        


st.title("3D Object Viewer from Point Cloud")

# Upload image input
template_file = st.file_uploader("Upload a Template Image", type=["jpg", "jpeg", "png"])
test_file = st.file_uploader("Upload a Test Image", type=["jpg", "jpeg", "png"])

if template_file is not None and test_file is not None:
    template_image = Image.open(template_file)
    st.subheader("Template Image")
    st.image(template_image, caption="Uploaded Template Image", use_column_width=True)

    test_image = Image.open(test_file)
    st.subheader("Test Image")
    st.image(test_image, caption="Uploaded Test Image", use_column_width=True)

    template_array = np.array(template_image)
    test_array = np.array(test_image)
    try:
    # Preprocess images (convert to grayscale)
        template = preprocess_image(template_array)
        test_img = preprocess_image(test_array)

        
        listHits = matchTemplates([("display", template)], test_img, score_threshold=0.5, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0)


        st.write("Template Matching Results")
        st.write(f"Found {len(listHits)} hits")

        for box in listHits:
            score = box[2]
            (x, y, w, h) = box[1]
            if score >0.7:
                cv2.rectangle(test_array, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(test_array, f"{score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 255, 0), 1, cv2.LINE_AA)


            if not os.path.exists('results'):
                os.makedirs('results')
            
            Image.fromarray(test_array).save(os.path.join('results', 'result.png'))

        
        st.subheader("Result")
        st.image(test_array, caption="Result Image", use_column_width=True)
        
    except Exception as e:
        st.error(f"Error processing the image: {e}")
        
        