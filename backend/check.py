import cv2
import os

img_folder = "dataset/train/images"
label_folder = "dataset/train/labels"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

for img in os.listdir(img_folder):

    path = img_folder + "/" + img
    label = label_folder + "/" + img.replace(".jpg", ".txt")

    image = cv2.imread(path)

    if image is None:
        print("❌ Error:", img)
        continue

    h, w, _ = image.shape

    if os.path.exists(label):
        with open(label) as f:
            for line in f:
                c, x, y, bw, bh = map(float, line.split())

                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                cv2.rectangle(image, (x1,y1), (x2,y2), (0,255,0), 2)

    # SAVE image instead of showing
    save_path = os.path.join(output_folder, img)
    cv2.imwrite(save_path, image)

print("✅ Done! Check 'output' folder")
