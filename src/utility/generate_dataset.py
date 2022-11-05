import scipy


def load_dataset(filename):
    mat_file = scipy.io.loadmat(filename)
    annotations = mat_file['RELEASE'][0]['annolist']
    acts = mat_file['RELEASE'][0]['act']

    image_file_names = [anno[0][0]['name'][0] for anno in annotations[0][0]['image']]

    img_train = mat_file['RELEASE'][0]['img_train'][0].T

    act_names = [act['act_name'][0][0] if act['act_name'][0].size > 0 else None for act in acts[0]]

    cat_names = [act['cat_name'][0][0] if act['cat_name'][0].size > 0 else None for act in acts[0]]

    person_count = [people[0].size for people in mat_file['RELEASE'][0]['single_person'][0]]

    result = []
    n = annotations[0].size

    for i in range(n):
        result.append([image_file_names[i], act_names[i], cat_names[i], person_count[i], img_train[i][0]])

    return result
