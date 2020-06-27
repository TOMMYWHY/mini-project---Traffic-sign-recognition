import os
import numpy as np
from collections import Counter
import sys


from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

classes = ['00000', '00001', '00002', '00003', '00004', '00005', '00006', '00007', '00008', '00009',
           '00010', '00011', '00012', '00013', '00014', '00015', '00016', '00017', '00018', '00019',
           '00020', '00021', '00022', '00023', '00024', '00025', '00026', '00027', '00028', '00029',
           '00030', '00031', '00032', '00033', '00034', '00035', '00036', '00037', '00038', '00039',
           '00040', '00041', '00042']
label_map = {}
for i in range(43):
    label_map[classes[i]] = i

source_dir = './Hue_features'

clf_type = 'rf'

def load_data():

    # Load original dataset
    f_train = np.load(os.path.join(source_dir, 'features_train.npy'))
    f_test = np.load(os.path.join(source_dir, 'features_test.npy'))
    l_train = np.load(os.path.join(source_dir, 'labels_train.npy'))
    l_test = np.load(os.path.join(source_dir,  'labels_test.npy'))
    print('Taking {} samples to train and {} samples to test the classifier.'.format(f_train.shape[0], f_test.shape[0]))
    return f_train, l_train, f_test, l_test

def preproc(f_train, f_test):

    # Build data Normalizer and
    sc = StandardScaler()
    sc.fit(f_train)

    # Normalize dataset with the same 'mean' and 'std'
    f_train_std = sc.transform(f_train)
    f_test_std = sc.transform(f_test)
    print('Training and validation data normalized.')

    return f_train_std, f_test_std

def train(f_train, l_train, mode_dir, classifier_type='svm'):

    # Build classifier
    if classifier_type == 'rf':
        clf = RandomForestClassifier(n_estimators=4, max_features='sqrt', n_jobs=-1)
        print('Random Forest classifier initialized.')
    elif classifier_type == 'ab':
        clf = AdaBoostClassifier(n_estimators=4, base_estimator=DecisionTreeClassifier(max_features='sqrt'))
        print('Ada boost classifier initialized.')
    print(clf)

    # Perform training
    print('Start training ...')
    clf.fit(f_train, l_train)
    print('Training Done.')

    # Save model to file
    joblib.dump(clf, mode_dir)
    print('Classifier model file saved to {}.'.format(mode_dir))

def validation(f_val, l_val, model_dir):

    # Load model
    clf = joblib.load(model_dir)
    print('Load classifier from {}'.format(model_dir))

    # Perform prediction
    print('{} samples to be validated ...'.format(f_val.shape[0]))
    preds = clf.predict(f_val)
    # probs = clf.predict_proba(f_val)

    # Calculate accuracy and confusion matrix
    accuracy = accuracy_score(l_val, preds)
    cfs_mtrx = confusion_matrix(l_val, preds)
    print('Validation done.')
    print('ACC: {}'.format(accuracy))
    # print('Confusion Matrix: {}'.format(cfs_mtrx))

    return accuracy, cfs_mtrx, preds

def main(argv):
    print("clf_type-----", argv[1])
    clf_type = argv[1]
    # Data loading
    features_train, labels_train, features_test, labels_test = load_data()

    # Data preprocessing
    features_train, features_test = preproc(features_train, features_test)


    if clf_type == 'rf':
        model_path = './ml_models/RandomForestClassifier_model.m'
    elif clf_type == 'ab':
        model_path = './ml_models/AdaBoostClassifier_model.m'


    if not os.path.exists(model_path):
        # Training
        train(features_train, labels_train, model_path, classifier_type=clf_type)

    # Validating
    acc, cf_mtrx, preds = validation(features_test, labels_test, model_path)

    print('\n ----------------------')
    # Report validating result
    print(classification_report(labels_test, preds, target_names=classes))


if __name__ == '__main__':
    main(sys.argv)
