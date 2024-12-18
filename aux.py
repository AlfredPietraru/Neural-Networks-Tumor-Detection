from torch.utils.data import Dataset
import os
import numpy as np
import cv2
from random import shuffle
import seaborn as sns
import matplotlib.pyplot as plt


TRAINING_DIR = "./Training"
TESTING_DIR = "./Testing"
GLIOMA_TUMOR = "glioma_tumor"
MENINGIOMA_TUMOR = "meningioma_tumor"
NO_TUMOR = "no_tumor"
PITUITARY_TUMOR = "pituitary_tumor"
VALIDATION_AMOUNT = 0.2
LABEL_LIST = {
    GLIOMA_TUMOR: 0,
    MENINGIOMA_TUMOR: 1,
    NO_TUMOR: 2,
    PITUITARY_TUMOR: 3
}

class ImagesMRIDataset(Dataset):
    def __init__(self, data, transformations = None):
        self.data = data
        self.transformations = transformations

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transformations == None:
            return cv2.imread(self.data[index][0]), self.data[index][1]
        return self.transformations(cv2.imread(self.data[index][0])), self.data[index][1]
        
def get_images_labels(img_dir):
    images_path = []
    labels = []
    for directory in LABEL_LIST.keys():
            current_path = img_dir + "/" + directory
            filenames = os.listdir(current_path)
            images_path += map(lambda x : current_path + "/" + x, filenames)
            labels += [LABEL_LIST[directory]] * len(filenames)
    result = list(zip(images_path, labels))
    shuffle(result)
    return result

def split_traing_data(final_info : list[tuple], validation : float):
    if (validation < 0.0 or validation > 1.0):
        print("Percent should be in range 0 - 1")
        exit(1)
    training_data = []
    validation_data = []
    for directory in LABEL_LIST.keys():
        current = list(filter(lambda x : x[1] == LABEL_LIST[directory], final_info))
        amount_training : int = int(len(current) * (1 - validation))
        training_data += current[:amount_training]
        validation_data += current[amount_training:]
    return training_data, validation_data

def get_data_distribution(data : list[tuple]):
    label_numbers = np.zeros(shape=(len(LABEL_LIST),))
    for label in LABEL_LIST.keys():
        idx = LABEL_LIST[label]
        label_numbers[idx] = len(list(filter(lambda x : x[1] == idx, data)))
    return label_numbers


def balance_no_tumor_class(path_image_label_info : list[tuple]):
    distribution = get_data_distribution(path_image_label_info)
    no_tumor_idx = LABEL_LIST[NO_TUMOR]
    no_tumor : list[tuple] = list(filter(lambda x : x[1] == no_tumor_idx, path_image_label_info))
    highest_number_class_index = np.argmax(distribution)
    how_many_more = distribution[highest_number_class_index] - 2 * len(no_tumor)
    result = path_image_label_info + no_tumor
    shuffle(no_tumor)
    result += no_tumor[0:int(how_many_more)]
    shuffle(result)
    return result

def get_training_testing_data(balanced : bool):
    training_info =  get_images_labels(TRAINING_DIR)
    test_info = get_images_labels(TESTING_DIR)
    if (balanced):
        return balance_no_tumor_class(training_info), test_info
    return training_info, test_info

def split_for_cross_validation(final_info: list[tuple], number_splits : int):
    validation_split = 1 / number_splits
    chunk_values = []
    for i in range(number_splits):
        validation_split = 1 / (number_splits - i)
        final_info, chunk = split_traing_data(final_info, validation_split)
        chunk_values.append(chunk)
    return chunk_values


def plot_data(training_loss, validation_loss, training_accuracy, validation_accuracy):
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
  sns.lineplot(x=range(1, len(training_loss) + 1), y=training_loss, ax=ax1, label='Training Loss', marker='o')
  sns.lineplot(x=range(1, len(validation_loss) + 1), y=validation_loss, ax=ax1, label='Validation Loss', marker='o')
  ax1.set_xlabel("Epoch")
  ax1.set_ylabel("Loss")
  ax1.set_title("Training and Validation Loss Over Epochs")
  ax1.legend()

  sns.lineplot(x=range(1, len(training_accuracy) + 1), y=training_accuracy, ax=ax2, label='Training Accuracy', marker='o')
  sns.lineplot(x=range(1, len(validation_accuracy) + 1), y=validation_accuracy, ax=ax2, label='Validation Accuracy', marker='o')
  ax2.set_xlabel("Epoch")
  ax2.set_ylabel("Accuracy")
  ax2.set_title("Training and Validation Accuracy Over Epochs")
  ax2.legend()

  plt.tight_layout()
  plt.show()