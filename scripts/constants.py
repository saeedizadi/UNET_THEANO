# Work space directory
HOME_DIR = '/grad/1/saeedi/'

# Path to raw data
pathToImages = '/local-scratch/saeedI/CVC-300/bbdd'
pathToMaps = '/local-scratch/saeedI/CVC-300/gtpolyp'
#pathToImagesNoAugment = '/home/titan/Saeed/saliency-salgan-2017/data/train_img_cross'
#pathToMapsNoAugment = '/home/titan/Saeed/saliency-salgan-2017/data/train_mask_cross'
pathToResMaps = '/home/saeedi/Projects/GAN_SKIN/data/results'
pathToFixationMaps = ''

# Path to processed data
pathOutputImages = '/local-scratch/saeedI/data/image320x240'
pathOutputMaps = '/local-scratch/saeedI/data/mask320x240'
pathToPickle = '/local-scratch/saeedI/data/pickle320x240'

# Path to pickles which contains processed data
TRAIN_DATA_DIR = '/local-scratch/saeedI/CVC-300/data/trainData.pickle'
TRAIN_DATA_DIR_CROSS = '/home/titan/Saeed/saliency-salgan-2017/data/pickle320x240/trainDataNoAugment.pickle'
VAL_DATA_DIR = '/local-scratch/saeedI/CVC-300/data/validationData.pickle'
TEST_DATA_DIR = '/home/saeedi/Projects/GAN_SKIN/data/pickle320x240/testData.pickle'

# Path to vgg16 pre-trained weights
PATH_TO_VGG16_WEIGHTS = '/home/saeedi/Projects/GAN_SKIN/models/vgg16.pkl'

# Input image and saliency map size
INPUT_SIZE = (320,240)

# Directory to keep snapshots
DIR_TO_SAVE = '../weights'
FIG_SAVE_DIR = '../figs'

#Path to test images
pathToTestImages = '/home/titan/Saeed/saliency-salgan-2017/images'
