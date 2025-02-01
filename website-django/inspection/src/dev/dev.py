import cv2, os, numpy as np, matplotlib.pyplot as plt
from pathlib import Path



def load_image(path):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Gambar tidak ditemukan di {path}")
    return image


def apply_gaussian_blur(image, kernel_size=(5, 5), sigma=1.4):
    return cv2.GaussianBlur(image, kernel_size, sigma)


def apply_canny(image, threshold1=50, threshold2=150):
    return cv2.Canny(image, threshold1, threshold2)


def apply_sobel(image, ksize=3):
    sobelx = cv2.Sobel(image, cv2.CV_64F, dx=1, dy=0, ksize=ksize)
    sobely = cv2.Sobel(image, cv2.CV_64F, dx=0, dy=1, ksize=ksize)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    magnitude = np.uint8(magnitude / magnitude.max() * 255)
    return sobelx, sobely, magnitude


def apply_laplacian(image, ksize=3):
    laplacian = cv2.Laplacian(image, cv2.CV_64F, ksize=ksize)
    laplacian_abs = np.uint8(np.absolute(laplacian))
    return laplacian_abs


def display_results(original, blurred, edges_canny, sobelx, sobely, magnitude, laplacian):
    plt.figure(figsize=(18, 12))

    plt.subplot(2, 4, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Gambar Asli")
    plt.axis("off")

    plt.subplot(2, 4, 2)
    plt.imshow(cv2.cvtColor(blurred, cv2.COLOR_BGR2RGB))
    plt.title("Gaussian Blur")
    plt.axis("off")

    plt.subplot(2, 4, 3)
    plt.imshow(edges_canny, cmap="gray")
    plt.title("Canny Edge Detection")
    plt.axis("off")

    plt.subplot(2, 4, 4)
    plt.imshow(magnitude, cmap="gray")
    plt.title("Sobel Magnitude")
    plt.axis("off")

    plt.subplot(2, 4, 5)
    plt.imshow(sobelx, cmap="gray")
    plt.title("Sobel X")
    plt.axis("off")

    plt.subplot(2, 4, 6)
    plt.imshow(sobely, cmap="gray")
    plt.title("Sobel Y")
    plt.axis("off")

    plt.subplot(2, 4, 7)
    plt.imshow(laplacian, cmap="gray")
    plt.title("Laplacian Edge Detection")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def main():
    script_dir = Path(__file__).resolve().parent.parent.parent
    image_dir = os.path.join(script_dir, "static/images/test")
    image_path = os.path.join(image_dir,"img1.jpg")
    original = load_image(image_path)
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = apply_gaussian_blur(original)
    gray_blurred = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges_canny = apply_canny(gray_blurred)
    sobelx, sobely, magnitude = apply_sobel(gray_blurred)
    laplacian = apply_laplacian(gray_blurred)
    display_results(original, blurred, edges_canny, sobelx, sobely, magnitude, laplacian)


if __name__ == "__main__":
    main()
