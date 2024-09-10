import cv2


def display_image(img, name='Image'):

    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()