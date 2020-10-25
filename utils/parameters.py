MODEL_CONFIG = 'configs/faster_rcnn_R_50_FPN.yaml'

ORGAN_CLASSES = ('leaf', 'flower', 'fruit', 'seed', 'stem', 'root')
ORGAN_LIST = ['leaf', 'flower', 'fruit', 'seed', 'stem', 'root']

HRBParis_imgDir = 'dataset/HerbarParis/scans/'
HRBParis_annoDir = 'dataset/HerbarParis/annotations/'
HRBParis_urls = 'dataset/HerbarParis/mnhn_urls.csv'
HRBParis_evalDir = 'dataset/HerbarParis/val/'
HRBParis_Dir = 'dataset/HerbarParis/'

HRBFR_imgDir = 'dataset/HerbarFR/scans/'
HRBFR_annoDir = 'dataset/HerbarFR/annotations/'
HRBFR_Dir = 'dataset/HerbarFR/'


extracted_Dir = 'dataset/Dataset/'
extracted_hrbDir = 'dataset/Dataset/Annotations/'
extracted_detections = 'dataset/Dataset/Detections/'

HRBFR_detectionsDir = 'detections/FR_Detections/'

BOUNDING_BOX_COLORS = [(0, 0, 255), (128, 0, 0), (255, 0, 255),
                       (255, 255, 0), (0, 128, 0), (128, 128, 128)]

HEIGHT = 300 * 4
WIDTH = 200 * 4

TOTAL_HRBParis_urls = 653
