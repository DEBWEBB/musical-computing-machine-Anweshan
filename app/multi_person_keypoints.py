# multi_person_keypoints.py
# Template for multi-person keypoint detection using torchvision's Keypoint R-CNN.
import cv2, argparse, time, json, os
import torch, torchvision
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0", help="0 or path to video")
parser.add_argument("--out", default="app/outputs", help="output folder")
parser.add_argument("--score_thresh", type=float, default=0.9)
args = parser.parse_args()

SRC = 0 if args.source == "0" else args.source
OUT = args.out
os.makedirs(OUT, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

cap = cv2.VideoCapture(SRC)
if not cap.isOpened():
    raise RuntimeError("Cannot open video source")
frame_idx = 0
frames_data = []

print("Processing multi-person. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    img_t = torch.tensor(img).permute(2,0,1).float().to(device)
    with torch.no_grad():
        outputs = model([img_t])[0]

    boxes = outputs['boxes'].cpu().numpy()
    scores = outputs['scores'].cpu().numpy()
    keypoints = outputs['keypoints'].cpu().numpy()
    frame_records = {"frame": frame_idx, "people": []}
    for i, sc in enumerate(scores):
        if sc < args.score_thresh:
            continue
        b = boxes[i].astype(int).tolist()
        kps = keypoints[i].tolist()
        person = {"score": float(sc), "box": b, "keypoints": kps}
        frame_records["people"].append(person)
        cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0,255,0), 2)
        for kp in kps:
            x,y,v = kp
            cv2.circle(frame, (int(x), int(y)), 3, (0,0,255), -1)
    frames_data.append(frame_records)
    cv2.imshow("Multi Person Keypoints", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ts = int(time.time())
outp = os.path.join(OUT, f"multi_keypoints_{ts}.json")
with open(outp, "w") as fh:
    json.dump(frames_data, fh)
print("Saved multi-person keypoints to", outp)
cap.release()
cv2.destroyAllWindows()