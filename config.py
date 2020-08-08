# encoding = utf-8

# Configuration of Active Learning on Hyper-Kvasir Labeled Dataset
# CANDIDATE_ROOT = '../image_dataset/split0'
# PATCH_ROOT = '../image_dataset/patches0'
# TEST_ROOT = '../image_dataset/split1'

# Configuration of Active Learning on MIT indoor scene classification dataset subset
CANDIDATE_ROOT = '../indoor_scene_dataset/train'
PATCH_ROOT = '../indoor_scene_dataset/train_patches'
TEST_ROOT = '../indoor_scene_dataset/test'

# STRATEGY = 'hybrid'
STRATEGY = 'random'
# STRATEGY = 'diversity'
# STRATEGY = 'loss'
# STRATEGY = 'entropy'

# Hyper-parameters
SEED = 2020
SUBSET = 500
BATCH = 12
K = 100              # numbers of request samples in each loop
CYCLES = 10
LR_classifier = 1e-3
LR_module = 1e-3
MOMENTUM = 0.9
WDECAY = 5e-4
MILESTONE = [15]
MARGIN = 0.1
# MARGIN = 1.0
# if STRATEGY is not 'hybrid' and STRATEGY is not 'loss' then WEIGHT = 0.0
# WEIGHT = 0.1
EPOCH = 20
# if STRATEGY is not 'hybrid' and STRATEGY is not 'loss' then EPOCHL = 0
EPOCHL = 0

# Because of the arbitrary order of os.listdir()
# A dictionary is needed to make sure each category is associated with the same index
# CATEGORY_MAPPING = {'barretts': 0, 'barretts-short-segment': 1, 'bbps-0-1': 2,
#                     'bbps-2-3': 3, 'cecum': 4, 'dyed-lifted-polyps': 5,
#                     'dyed-resection-margins': 6, 'esophagitis-a': 7, 'esophagitis-b-d': 8,
#                     'hemorrhoids': 9, 'ileum': 10, 'impacted-stool': 11,
#                     'normal-z-line': 12, 'polyps': 13, 'pylorus': 14,
#                     'retroflex-rectum': 15, 'retroflex-stomach': 16, 'ulcerative-colitis-0-1': 17,
#                     'ulcerative-colitis-1-2': 18, 'ulcerative-colitis-2-3': 19, 'ulcerative-colitis-grade-1': 20,
#                     'ulcerative-colitis-grade-2': 21, 'ulcerative-colitis-grade-3': 22}

# category mapping for indoor scene classification
CATEGORY_MAPPING = {'airport_inside': 0, 'artstudio': 1, 'auditorium': 2,
                    'bakery': 3, 'bar': 4, 'bathroom': 5, 'bedroom': 6,
                    'bookstore': 7, 'bowling': 8, 'buffet': 9, 'casino': 10,
                    'children_room': 11, 'church_inside': 12, 'classroom': 13,
                    'cloister': 14, 'closet': 15, 'clothingstore': 16,
                    'computerroom': 17, 'concert_hall': 18, 'corridor': 19}

# Configuration of Active Learning on the Subset of Hyper-Kvasir Labeled Dataset
# CANDIDATE_ROOT = 'unlabel'
# PATCH_ROOT = 'patch'
# TEST_ROOT = 'test'
#
# SEED = 2020
# BATCH = 10
# K = 12              # numbers of request samples in each loop
# CYCLES = 10
# LR = 1e-2
# MOMENTUM = 0.9
# WDECAY = 5e-4
# MILESTONE = [15]
# MARGIN = 1.0
# WEIGHT = 1.0
# EPOCH = 20
# EPOCHL = 15
#
# # Because of the arbitrary order of os.listdir()
# # A dictionary is needed to make sure each category is associated with the same index
# CATEGORY_MAPPING = {'bbps-2-3': 0, 'cecum': 1, 'dyed-resection-margins': 2,
#                     'normal-z-line': 3, 'polyps': 4, 'pylorus': 5}




