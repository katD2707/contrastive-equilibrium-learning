# Frist, run "bash makelist_path.sh" to create path_vox2.txt
# Then run "python makelist_post.py"

import numpy as np
import glob

def generate_label(label):
    label_unique = list(set(label))
    label_unique.sort()
    label_to_num = []
    for i, v in enumerate(label):
        label_to_num.append(label_unique.index(v))
    return np.array(label_to_num)

# start loading
#spk = []
file = []
path_to_dts = '../youtube-dataset/**'
lang_vids = glob.glob(path_to_dts, recursive=True)

for path in lang_vids:
    check_path = path.split('\\')
    if len(check_path) > 3:
        file.append('/'.join(check_path[1:]))

#spk_to_num = generate_label(spk)
#print(len(np.unique(spk_to_num)))
# print(len(file))
#
output = open('train_youtube.txt','w')
for i in range(len(file)):
    output.write(file[i]+'\n')
output.close()
