from ultralytics.data.annotator import auto_annotate

auto_annotate(data="/home/skorp321/Projects/test/data/vid1", det_model="yolo11x.pt", sam_model="sam2_b.pt")