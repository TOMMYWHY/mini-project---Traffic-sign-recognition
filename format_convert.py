import numpy as np
import os
import sys

classes = ['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009',
           '00010', '00011', '00012', '00013', '00014', '00015', '00016', '00017', '00018', '00019',
           '00020', '00021', '00022', '00023', '00024', '00025', '00026', '00027', '00028', '00029',
           '00030', '00031', '00032', '00033', '00034', '00035', '00036', '00037', '00038', '00039',
           '00040', '00041', '00042']
label_map = {}
for i in range(43):
    label_map[classes[i]] = i

# roots = {'train': './GTSRB_Final_Training_Hue/HueHist', 'test': './GTSRB_Online_Test_Hue/HueHist'}
roots = {'train': "", 'test': ""}

def convert(sub):
    root = roots[sub]
    label_strs = os.listdir(root)

    features = []
    labels = []
    num_images = 0
    for label_str in label_strs:
        label = label_map[label_str]

        feature_root = os.path.join(root, label_str)

        feature_txts = os.listdir(feature_root)
        num_images += len(feature_txts)

        for feature_txt in feature_txts:
            with open(os.path.join(feature_root, feature_txt), 'r') as f:
                lines = f.readlines()
                feature = np.asarray([float(x.strip()) for x in lines])
                features.append(feature)
                labels.append(label)

    features = np.vstack(features)
    labels = np.asarray(labels)
    print('total number images:', num_images, 'total number features:', features.shape, 'total number labels:', labels.shape)

    np.save('./Hue_features/features_' + sub + '.npy', features)
    np.save('./Hue_features/labels_' + sub + '.npy', labels)


def main(argv):
    roots['train'] =argv[1] + '/HueHist'
    roots['test'] =argv[2] + '/HueHist'
    convert('train')
    convert('test')



if __name__ == '__main__':
    main(sys.argv)
