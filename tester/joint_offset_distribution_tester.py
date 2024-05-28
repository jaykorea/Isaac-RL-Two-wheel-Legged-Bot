import matplotlib.pyplot as plt
import numpy as np

def get_offset_distribution_plot():
    #* TWBR Joint limits From here*#
    left_hip_joint_lower_limit = -0.383972
    left_hip_joint_upper_limit = 0.7854
    left_shoulder_joint_lower_limit = 0
    left_shoulder_joint_upper_limit = 1.5708
    left_leg_joint_lower_limit = 0.
    left_leg_joint_upper_limit = 1.5708

    right_hip_joint_lower_limit = -0.7854
    right_hip_joint_upper_limit = 0.383972
    right_shoulder_joint_lower_limit = 0
    right_shoulder_joint_upper_limit = 1.5708
    right_leg_joint_lower_limit = 0.
    right_leg_joint_upper_limit = 1.5708
    #* TWBR Joint limits To here*#

    #* TWBR Joint deviation From here*#
    std_dev_l_hip = (left_hip_joint_upper_limit - left_hip_joint_lower_limit) / 256  # 99.7% should fall within +/- 3 std devs
    std_dev_l_shoulder = (left_shoulder_joint_upper_limit - left_shoulder_joint_lower_limit) / 6
    std_dev_l_leg = (left_leg_joint_upper_limit - left_leg_joint_lower_limit) / 12
    std_dev_r_hip = (right_hip_joint_upper_limit - right_hip_joint_lower_limit) / 256
    std_dev_r_shoulder = (right_shoulder_joint_upper_limit - right_shoulder_joint_lower_limit) / 6
    std_dev_r_leg = (right_leg_joint_upper_limit - right_leg_joint_lower_limit) / 12
    #* TWBR Joint deviation To here*#

    mean_l_hip = (left_hip_joint_lower_limit + left_hip_joint_upper_limit) / 2 - 0.200714
    mean_l_shoulder = (left_shoulder_joint_lower_limit + left_shoulder_joint_upper_limit) / 2 - 0.7854
    mean_l_leg = (left_leg_joint_lower_limit + left_leg_joint_upper_limit) / 2 - 0.7854

    mean_r_hip = (right_hip_joint_lower_limit + right_hip_joint_upper_limit) / 2 + 0.200714
    mean_r_shoulder =(right_shoulder_joint_lower_limit + right_shoulder_joint_upper_limit) / 2 - 0.7854
    mean_r_leg = (right_leg_joint_lower_limit + right_leg_joint_upper_limit) / 2 - 0.7854
    
    print("mean left: ", mean_l_shoulder)
    print("mean right: ", mean_r_shoulder)
    
    # Generating new samples from the updated Gaussian distributions
    samples_l_hip = np.random.normal(mean_l_hip, std_dev_l_hip, 10000)
    samples_l_shoulder = np.random.normal(mean_l_shoulder, std_dev_l_shoulder, 10000)
    samples_l_leg = np.random.normal(mean_l_leg, std_dev_l_leg, 10000)

    samples_r_hip = np.random.normal(mean_r_hip, std_dev_r_hip, 10000)
    samples_r_shoulder = np.random.normal(mean_r_shoulder, std_dev_r_shoulder, 10000)
    samples_r_leg = np.random.normal(mean_r_leg, std_dev_r_leg, 10000)

    # Replotting the distributions with updated means
    fig, ax = plt.subplots(3, 2, figsize=(12, 9))
    ax[0, 0].hist(samples_l_hip, bins=150, density=True, color='blue', alpha=0.7)
    ax[0, 0].set_title('Left Hip Position Offset')

    ax[0, 1].hist(samples_r_hip, bins=150, density=True, color='blue', alpha=0.7)
    ax[0, 1].set_title('Right Hip Position Offset')

    ax[1, 0].hist(samples_l_shoulder, bins=150, density=True, color='blue', alpha=0.7)
    ax[1, 0].set_title('Left Shoulder Position Offset')

    ax[1, 1].hist(samples_r_shoulder, bins=150, density=True, color='blue', alpha=0.7)
    ax[1, 1].set_title('Right Shoulder Position Offset')

    ax[2, 0].hist(samples_l_leg, bins=150, density=True, color='blue', alpha=0.7)
    ax[2, 0].set_title('Left Leg Position Offset')

    ax[2, 1].hist(samples_r_leg, bins=150, density=True, color='blue', alpha=0.7)
    ax[2, 1].set_title('Right Leg Position Offset')

    plt.tight_layout()
    plt.show()

def main():
    get_offset_distribution_plot()
    
if __name__ == "__main__":
    main()

