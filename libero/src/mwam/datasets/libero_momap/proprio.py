"""Single training/inference definition of current LIBERO proprioception."""
import numpy as np


def official_observation_proprio(observation):
    pos=np.asarray(observation['robot0_eef_pos'],np.float64)
    quat=np.asarray(observation['robot0_eef_quat'],np.float64).copy() # xyzw
    grip=np.asarray(observation['robot0_gripper_qpos'],np.float64)
    if pos.shape!=(3,) or quat.shape!=(4,) or grip.shape!=(2,):raise ValueError('Expected pos3, xyzw4, gripper2')
    if not all(np.isfinite(v).all() for v in (pos,quat,grip)) or not np.isclose(np.linalg.norm(quat),1,atol=1e-5):raise ValueError('Invalid observation')
    w=np.clip(quat[3],-1,1);den=np.sqrt(1-w*w)
    rot=np.zeros(3) if den==0 else quat[:3]*2*np.arccos(w)/den
    return np.r_[pos,rot,grip].astype(np.float32)


def proprio_from_saved_state(model,data):
    """Call after mj_setState + mj_forward. Preserve the official body sign."""
    fingers=[model.joint('gripper0_finger_joint1').id,model.joint('gripper0_finger_joint2').id]
    return official_observation_proprio(dict(robot0_eef_pos=data.site_xpos[model.site('gripper0_grip_site').id],
        robot0_eef_quat=data.xquat[model.body('robot0_right_hand').id][[1,2,3,0]],
        robot0_gripper_qpos=data.qpos[model.jnt_qposadr[fingers]]))
