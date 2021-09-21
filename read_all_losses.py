import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_directory_from_id_openmind(root, id:int):
    kinect_set = root + "/Kinect"
    kinects = ["/Kin01", "/Kin02"]
    subjects = ["/S01", "/S02", "/S03", "/S04", "/S05", "/S06", "/S07", "/S08", "/S09", "/S10", "/S11", "/S12"]
    actions = ["/A01", "/A02", "/A03", "/A04", "/A05", "/A06", "/A07", "/A08"]
    reps = ["/R01", "/R02", "/R03", "/R04", "/R05"]

    n_reps = len(reps)          # 5
    n_actions = len(actions)    # 8
    n_subjects = len(subjects)  # 12
    n_kinects = len(kinects)    # 2

    rep = reps[id % n_reps]
    id //= n_reps
    action = actions[id % n_actions]
    id //= n_actions
    subject = subjects[id % n_subjects]
    id //= n_subjects
    kinect = kinects[id % n_kinects]
    id //= n_kinects
    assert id == 0  # This confirms that the original value of id was within range

    return kinect_set + kinect + subject + action + rep + "/"

directory = get_directory_from_id_openmind(os.getcwd(), 1)
# losscurvename = '/Users/handongwang/modular/neurogym/multitask/files/losscurve.npy'
losscurvename = directory + 'mask_losscurve.npy'
data = np.load(losscurvename)
num_directories = int(sys.argv[1])
directory_count = num_directories
for i in range(2, num_directories):
    directory = get_directory_from_id_openmind(os.getcwd(), i)
    # losscurvename = '/Users/handongwang/modular/neurogym/multitask/files/losscurve.npy'
    losscurvename = directory + 'mask_losscurve.npy'
    try:
        directory_data = np.load(losscurvename)
        data += directory_data
    except FileNotFoundError:
        directory_count -= 1
data /= directory_count
print(data)
print("directory_count", directory_count)
plt.plot(data)
plt.savefig("mask_losscurve.png")
plt.show()