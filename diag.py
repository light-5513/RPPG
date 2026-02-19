import os, cv2, pickle, numpy as np

raw_path = 'datasets/ubfc-rppg'
subjects = sorted(os.listdir(raw_path))[:3]
for subj in subjects:
    subj_path = os.path.join(raw_path, subj)
    if not os.path.isdir(subj_path):
        continue
    files = os.listdir(subj_path)
    print(f'{subj}: {files}')
    for f in files:
        fp = os.path.join(subj_path, f)
        if f.endswith(('.avi','.mp4','.mkv')):
            cap = cv2.VideoCapture(fp)
            opened = cap.isOpened()
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f'  Video opened: {opened}, frames: {frames}, fps: {fps}')
            if opened:
                ret, frame = cap.read()
                print(f'  Read first frame: {ret}, shape: {frame.shape if ret else None}')
            cap.release()
        if f.endswith('.txt'):
            gt = np.loadtxt(fp)
            print(f'  GT shape: {gt.shape}')

for name in ['train_data.pkl', 'val_data.pkl']:
    pth = os.path.join('datasets/processed', name)
    if os.path.exists(pth):
        with open(pth, 'rb') as fh:
            d = pickle.load(fh)
        print(f'{name}: clips={len(d["clips"])}, labels={len(d["labels"])}')
