from MTM import matchTemplates
import cv2
import matplotlib.pyplot as plt


cropping = False
x1, y1, x2, y2 = 0, 0, 0, 0


def mouse_crop(event, x, y, flags, param):
    global x1, y1, x2, y2, cropping

    # Start cropping on left mouse button click
    if event == cv2.EVENT_LBUTTONDOWN:
        x1, y1 = x, y
        cropping = True

    # Update the rectangle while dragging the mouse
    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            temp_img = param.copy()
            cv2.rectangle(temp_img, (x1, y1), (x, y), (0, 255, 0), 2)
            cv2.imshow("Image", temp_img)

    # Finalize cropping on left button release
    elif event == cv2.EVENT_LBUTTONUP:
        x2, y2 = x, y
        cropping = False
        cv2.rectangle(param, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Image", param)

def main():
    image_path = "img/coins.jpg" 
    original_image = cv2.imread(image_path)

    if original_image is None:
        print("Error: Could not load the image.")
        return

    clone = original_image.copy()
    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_crop, param=clone)

    print("Instructions: Drag the mouse to select a region. Press 'c' to crop, 'r' to reset.")

    while True:
        cv2.imshow("Image", clone)
        key = cv2.waitKey(1) & 0xFF

        # Press 'c' to crop and save the selected region
        if key == ord("c"):
            if x1 != x2 and y1 != y2:
                cropped_image = original_image[y1:y2, x1:x2]
                print(f"Cropped region coordinates: (x1, y1) = ({x1}, {y1}), (x2, y2) = ({x2}, {y2})")

                # Save the cropped image
                cv2.imwrite("cropped_image.jpg", cropped_image)
                print("Cropped image saved as 'cropped_image.jpg'.")
                cv2.imshow("Cropped", cropped_image)
            else:
                print("Error: Invalid crop region.")

        # Press 'r' to reset the image
        elif key == ord("r"):
            clone = original_image.copy()

        # Press 'q' to quit
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
    
    
    image = cv2.imread(image_path, -1)
    template = cv2.imread('cropped_image.jpg', -1)

    if image is None:
        raise FileNotFoundError("Image file not found.")
    if template is None:
        raise FileNotFoundError("Template file not found.")

    listHits = matchTemplates([("display", template)], image, score_threshold=0.5, method=cv2.TM_CCOEFF_NORMED, maxOverlap=0)

    print("Found {} hits".format( len(listHits) ) )
    print("Hits: ", listHits)


    for box in listHits:
        score = box[2]
        (x, y, w, h) = box[1]
        if score > 0.7:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"{score:.2f}", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 
                        1, (0, 255, 0), 3, cv2.LINE_AA)

    # Display the image with rectangles
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    main()
