import cv2, torch, matplotlib.pyplot as plt

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to("cpu")
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imgbatch = transform(img).to("cpu")
    with torch.no_grad():
        prediction = midas(imgbatch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        output = prediction.cpu().numpy()
        print(output)
    plt.imshow(output)
    cv2.imshow("frame", frame)
    plt.pause(0.00001)
    if cv2.waitKey(1) & 0xFF == ord("n"):
        cap.release()
        cv2.destroyAllWindows()
plt.show()
