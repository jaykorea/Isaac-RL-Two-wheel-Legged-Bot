import numpy as np
import torch

def get_rotation_matrix(rot_x, rot_y, rot_z):
    """Create a rotation matrix from Euler angles."""
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(rot_x), -np.sin(rot_x)],
        [0, np.sin(rot_x), np.cos(rot_x)]
    ])
    ry = np.array([
        [np.cos(rot_y), 0, np.sin(rot_y)],
        [0, 1, 0],
        [-np.sin(rot_y), 0, np.cos(rot_y)]
    ])
    rz = np.array([
        [np.cos(rot_z), -np.sin(rot_z), 0],
        [np.sin(rot_z), np.cos(rot_z), 0],
        [0, 0, 1]
    ])
    return rz @ ry @ rx

def T(axis, l):
    if axis == 'x':
        rx = np.array([
            [1, 0, 0, l],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        return rx
    elif axis == 'y':
        ry = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, l],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        return ry
    elif axis == 'z':
        rz = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, l],
            [0, 0, 0, 1]])
        return rz
    
def R(axis, r):
    if axis == 'x':
        rx = np.array([
        [1, 0, 0, 0],
        [0, np.cos(r), -np.sin(r), 0],
        [0, np.sin(r), np.cos(r), 0],
        [0, 0, 0, 1]
    ])
        return rx
    elif axis == 'y':
        ry = np.array([
            [np.cos(r), 0, np.sin(r), 0],
            [0, 1, 0, 0],
            [-np.sin(r), 0, np.cos(r), 0],
            [0, 0, 0, 1]
        ])
        return ry
    elif axis == 'z':
        rz = np.array([
        [np.cos(r), -np.sin(r), 0, 0],
        [np.sin(r), np.cos(r), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
        ])
        return rz
    
def get_base_height(lz1, lz2, lz3, rz1, rz2, rz3):
    Debug_enable = True
    lr1 = np.deg2rad(lz1)
    lr2 = np.deg2rad(lz2)
    lr3 = np.deg2rad(lz3)

    rr1 = np.deg2rad(rz1)
    rr2 = np.deg2rad(rz2)
    rr3 = np.deg2rad(rz3)

    left_joint_trm =\
                    T('x', -0.0545) @ T('y', 0.08)  @ T('z', -0.0803) @ R('z', 1.57079) @ R('x', 1.57079) @\
                    R('z', lr1) @ T('x', 0.0265) @ T('z', 0.06) @ R('y',1.57079) @\
                    R('z', lr2) @ T('x', 0.22) @ T('z', 0.07005) @\
                    R('z', lr3) @ T('x', -0.18883) @ T('y', -0.11289) @ T('z', 0.039877)
                    #T('y', -0.053)

    right_joint_trm =\
                     T('x', -0.0545) @ T('y', -0.08) @ T('z', -0.0803) @ R('z', 1.57097) @ R('x', 1.57097)@\
                     R('z', rr1) @ T('x', -0.0265) @ T('z', 0.06) @ R('x', 3.14158)  @ R('y', -1.57079) @\
                     R('z', -rr2) @ T('x', 0.22) @ T('z', 0.07005) @\
                     R('z', -rr3) @ T('x', -0.18883) @ T('y', 0.11289) @ T('z', 0.039877)
                     #T('y', 0.053)

    
    left_joint_z_pos = left_joint_trm[2,3]
    right_joint_z_pos = right_joint_trm[2,3]

    base_height = -((left_joint_z_pos + right_joint_z_pos)/2) + 0.053

    if Debug_enable == True:
        print("Left")
        print(left_joint_trm)
        print("Right")
        print(right_joint_trm)
        print("")
        print("base_height: ", base_height)


    return base_height

def main():
    get_base_height(0, 0, 0, 0, 0, 0)
    #get_base_height(0, 14, 47, 0, 14, 47)
    #get_base_height_constant(0, -45, 45, 0, -45, 40)
    
if __name__ == "__main__":
    main()
