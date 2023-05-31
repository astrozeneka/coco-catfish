call conda activate base
set PYTHONUNBUFFERED=1
set PYTHONPATH=C:\Users\KU\Desktop\AGB\Ryan\projects\pytorch-draft-01\src\coco-fish-2\visiondetection
powershell "python 2_TrainDetectionOCR3.py -e %1 -r %2 -d %3 -o model-%1-%2-%3.pth | tee output/e-rcnn-%1-%2-%3.txt"