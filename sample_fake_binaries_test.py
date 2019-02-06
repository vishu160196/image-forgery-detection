from multiprocessing import Process, Manager
from imageio import imread
import pickle
import numpy as np
import cv2

fake_path = 'dataset-dist/phase-01/training/fake/'
pristine_path = 'dataset-dist/phase-01/training/pristine/'
mask_path = fake_path + 'masks/'

def count_255(mask):
    i=0
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if mask[row,col]==255:
                i+=1
    return i

def sample_fake(img, mask):
    kernel_size = 64
    stride = 8
    threshold=1024

    samples = []

    for y_start in range(0, img.shape[0] - kernel_size + 1, stride):
        for x_start in range(0, img.shape[1] - kernel_size + 1, stride):

            c_255 = count_255(mask[y_start:y_start + kernel_size, x_start:x_start + kernel_size])

            if (c_255 > threshold) and (kernel_size * kernel_size - c_255 > threshold):
                samples.append(img[y_start:y_start + kernel_size, x_start:x_start + kernel_size, :3])

    return samples


def process(batch, common_list, images, masks):

    for img, mask in zip(images, masks):
        samples=sample_fake(img, mask)
        for s in samples:
            common_list.append(s)
        print('Number of samples = ' + str(len(samples)))

    print(f'batch {batch} completed')

def main():

    with open('data/x_test.pickle', 'rb') as f:
        x_test=pickle.load(f)

    x_test_masks = []
    x_test_fake_images=[]
    for img_name in x_test:
        x_test_masks.append(imread(mask_path+img_name.split('.')[0]+'.mask.png'))
        x_test_fake_images.append(imread(fake_path+img_name))

    # Convert grayscale images to binary
    binaries=[]

    for grayscale in x_test_masks:
        blur = cv2.GaussianBlur(grayscale,(5,5),0)
        ret,th = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        binaries.append(th)

    m = Manager()
    common_list = m.list()

    processes = []
    for batch in range(9):
        processes.append(Process(target=process, args=(batch, common_list, x_test_fake_images[batch*10:(batch+1)*10],
                                                       binaries[batch*10:(batch+1)*10])))

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    samples_fake_np = common_list[0][np.newaxis, :, :, :]
    for fake_sample in common_list[1:]:
        samples_fake_np = np.concatenate((samples_fake_np, fake_sample[np.newaxis, :, :, :3]), axis=0)

    print('done')
    np.save('sample_fakes_np_test.npy', samples_fake_np)


if __name__ == '__main__':
    main()
