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

directory = get_directory_from_id_openmind(os.getcwd(), int(sys.argv[1]))

fig, axes = plt.subplots(2)
fig.suptitle('Loss and final loss: evaluating model trained on a single diirectoory')

# losscurvename = '/Users/handongwang/modular/neurogym/multitask/files/losscurve.npy'
losscurvename = directory + 'mask_eval_losses.npy'
data = np.load(losscurvename)
print(data)
axes[0].plot(data)
# axes[0].savefig("eval_losses.png")
# plt.show()
finallosscurvename = directory + 'mask_eval_final_losses.npy'
finaldata = np.load(finallosscurvename)
print(finaldata)
axes[1].plot(finaldata)
# axes[1].savefig("eval_final_losses.png")
plt.savefig("mask_eval_losses.png")
# plt.show()
