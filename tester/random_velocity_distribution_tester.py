import matplotlib.pyplot as plt
import numpy as np

def get_gaussian_velocity():
    x_limits_lower = -1.
    x_limits_upper = 1.
    y_limits_lower = -0
    y_limits_upper = 0.
    yaw_limits_lower = -1.
    yaw_limits_upper = 1.
    x_std_dev = 6
    y_std_dev = 6
    yaw_std_dev = 6 

    #* TWBR velocity range deviation -JH
    std_dev_x = (x_limits_upper - x_limits_lower) / x_std_dev  # 99.7% should fall within +/- 3 std devs
    std_dev_y = (y_limits_upper - y_limits_lower) / y_std_dev
    std_dev_yaw = (yaw_limits_upper - yaw_limits_lower) / yaw_std_dev
    mean_x = (x_limits_lower + x_limits_upper) / 2
    mean_y = (y_limits_lower + y_limits_upper) / 2
    mean_yaw = (yaw_limits_lower + yaw_limits_upper) / 2

    # 각 env_id에 대해 가우시안 랜덤 값을 생성
    gaussian_velocity_x_sample = np.random.normal(mean_x, std_dev_x, 10000)
    gaussian_velocity_y_sample = np.random.normal(mean_y, std_dev_y, 10000)
    gaussian_velocity_yaw_sample = np.random.normal(mean_yaw, std_dev_yaw, 10000)
    # Replotting the distributions with updated means
    fig, ax = plt.subplots(3, 1, figsize=(12, 9))
    ax[0].hist(gaussian_velocity_x_sample, bins=200, density=True, color='blue', alpha=0.7)
    ax[0].set_title('Gaussian X velocity')

    ax[1].hist(gaussian_velocity_y_sample, bins=200, density=True, color='blue', alpha=0.7)
    ax[1].set_title('Gaussian Y velocity')

    ax[2].hist(gaussian_velocity_yaw_sample, bins=200, density=True, color='blue', alpha=0.7)
    ax[2].set_title('Gaussian Yaw velocity')


    plt.tight_layout()
    plt.show()

def main():
    get_gaussian_velocity()
    
if __name__ == "__main__":
    main()

