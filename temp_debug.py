from importlib import util
import cv2
spec = util.spec_from_file_location('main_mod', 'targil-bait-1/Main_123456789.py')
mod = util.module_from_spec(spec)
spec.loader.exec_module(mod)
proc = mod.preprocess_image('D:/PycharmProjects/image-processing/targil-bait-1/image_1.png')
edge = mod.detect_edges_for_hough(proc)
cv2.imwrite('targil-bait-1/edge_debug.png', edge)
