# Program structure

```text
robot_agent/
├── main.py                     # Orchestrates the entire pipeline, handles streaming input and output.
├── config.py                   # System configurations (API keys, model paths, robot parameters).
├── requirements.txt            # Python dependencies.
├── perception/
│   ├── vlm_module.py           # Integrates VLM (e.g., OpenVLA, Octo inference for scene understanding).
│   ├── 3d_perception.py        # Handles depth/point cloud processing, object segmentation, pose estimation.
│   └── camera_interface.py     # Manages streaming camera input (e.g., using OpenCV or ROS image topics).
├── planning/
│   ├── llm_planner.py          # Interfaces with the chosen LLM (e.g., Llama-2 via vLLM), handles prompt engineering, generates high-level plans.
│   └── skill_orchestrator.py   # Parses LLM's plan (e.g., JSON, code), maps to modular skills, and manages their execution sequence.
├── skills/                     # Modular Skill Library (robot-agnostic or adaptable functions)
│   ├── manipulation_skills.py  # Functions for grasping, placing, pouring (e.g., `grasp(obj_id, pose)`).
│   ├── navigation_skills.py    # Functions for movement, path planning, obstacle avoidance (e.g., `move_to(coords)`).
│   ├── sensing_skills.py       # Functions to query perception module (e.g., `detect_object(name)`, `get_object_pose(id)`).
│   └── utility_skills.py       # General helper functions (e.g., `wait(duration)`).
├── control/
│   └── robot_interface.py      # Low-level interface to the humanoid robot's hardware/ROS topics (e.g., sending joint commands, gripper control).
├── feedback/
│   ├── monitor.py              # Monitors skill execution, detects failures (e.g., using visual, force feedback).
│   └── error_handler.py        # Formulates detected failures into structured feedback for LLM replanning.
├── data/                       # Directory for datasets, demonstration data, and logs.
└── models/                     # Directory for pre-trained VLM/LLM model checkpoints.
```
