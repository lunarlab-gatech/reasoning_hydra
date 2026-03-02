# <div align="center">Relationship-Aware Hierarchical 3D Scene Graph</div>

This package implements an **enhanced hierarchical 3D scene graph** based on [Hydra](https://github.com/MIT-SPARK/Hydra/tree/main), integrating open-vocabulary features for rooms and objects, and supporting object-relational reasoning.

We leverage a **Vision-Language Model (VLM)** to infer semantic relationships. Additionally, we introduce a **task reasoning module** that combines **Large Language Models (LLM)** and a VLM to interpret the scene graph’s semantic and relational information, enabling agents to reason about tasks and interact with their environment intelligently.

## Installation

### Docker Setup

Make sure to install:
- [Docker](https://docs.docker.com/engine/install/ubuntu/)

Then, create the ros workspace with the `src` folder, and then navigate the terminal to that `src` folder. _Creating the workspace outside the docker helps you keep your files and changes within the workspace even if you delete the un-committed docker container._ Then, clone this repository into the `src` folder.

After that, navigate to the `docker` directory. Log in to the user that you want the docker file to create in the container. Then, Edit the `enter_container.sh` script with the following paths:
- `DATA_DIR=`: The directory where the HERCULES dataset is located
- `WS_DIR=`: The directory of the ROS workspace

Now, run the following commands:
```
build_container.sh
run_container.sh
```

To re-enter the container, run the following command:
```
enter_container.sh
```

### Build

Build the repository in **Release mode**:

```bash
catkin init
catkin config -DCMAKE_BUILD_TYPE=Release
cd src
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
vcs import . < reasoning_hydra/install/packages.repos
rosdep update
rosdep install --from-paths . --ignore-src -r -y
cd ..
catkin build
```

### Python Environment for Semantics and Reasoning

Follow the instructions below (similar to instruction in [semantic_inference_ros](https://github.com/ntnu-arl/semantic_inference_ros)) to set up the Python environment required to run the semantic and reasoning models:

```bash
cd src/semantic_inference
git submodule add --force git@github.com:ntnu-arl/DeepSeek-VL2.git semantic_inference_python/src/semantic_inference_python/models/deepseek 
git submodule init
git submodule update --recursive
cd semantic_inference_python
python3.8 -m venv --system-site-packages ros_semantics_env
source ros_semantics_env/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## Usage

### Scene Graph Construction

The system supports multiple datasets and online deployment on robots with GPU capabilities (e.g., **Nvidia Jetson Orin AGX**).

#### Uhumans2

Download rosbags from [Uhumans2 dataset](https://web.mit.edu/sparklab/datasets/uHumans2/).

Start the scene graph:

```bash
source ~/reasoning_hydra_ws/devel/setup.bash
roslaunch hydra_ros uhumans2.launch
```

In a separate terminal, play the rosbag:

```bash
source ~/reasoning_hydra_ws/devel/setup.bash
rosbag play uHumans2_office_s1_00h.bag
```

#### Replica

Follow [NICE-SLAM instructions](https://github.com/cvg/nice-slam#replica-1) to download posed RGB-D data from Replica scenes.

Run the scene graph:

```bash
roslaunch hydra_ros replica.launch
```

Publish the data:

```bash
roslaunch hydra_ros publish_replica.launch dataset_path:=<Path to your replica dataset> scene_name:=<Scene name>
```

#### Habitat-Matterport 3D Semantics Dataset

Follow [HOV-SG instructions](https://github.com/hovsg/HOV-SG?tab=readme-ov-file#habitat-matterport-3d-semantics) (Step 2 can be skipped) to download posed RGB-D data from several scenes.

Run the scene graph:

```bash
roslaunch hydra_ros hm3dsem.launch
roslaunch hydra_ros publish_hm3dsem.launch dataset_path:=<Path to hm3d_trajectories> scene_name:=<Scene name>
```

#### Robot Deployment

To run the scene graph on your robot:

- Robot must provide posed RGB-D data as `sensor_msgs/Image`
- Pose must be provided via **TFs**

Update [robot.launch](https://github.com/ntnu-arl/reasoning_hydra_ros/blob/master/hydra_ros/launch/robot.launch) with the correct TFs and camera topic names, then run:

```bash
roslaunch hydra_ros robot.launch
```

We provide recorded data from experiments with an ANYMal robot. Download it [here](https://huggingface.co/datasets/ntnu-arl/reasoning-graph-dataset).

To use this data:

```bash
roslaunch hydra_ros robot.launch playback_mode:=True
```

Then play one of the downloaded rosbags:

```bash
rosbag play <bag_to_play> --topics /tf /camera/aligned_depth_to_color/image_raw/compressedDepth /camera/color/camera_info /camera/color/image_raw/compressed --clock
```

---

### Task Reasoning

The **reasoning module** (VLM + LLMs) requires an **internet connection**.

- LLM queries are done via **OpenAI API**.
- A large VLM is hosted externally (setup instructions: [semantic_inference_ros](https://github.com/ntnu-arl/semantic_inference_ros))

**IMPORTANT:** When using the reasoning module, set your OpenAI and FastAPI (see https://github.com/ntnu-arl/semantic_inference_ros) keys as environment variables before launching the ROS nodes:
```bash
export OPENAI_API_KEY=<Your OpenAI API Key>
export FASTAPI_API_KEY=<Your server FastAPI Key>
```

Once the scene graph is constructed, either:

1. Use the provided **rviz GUI** to interact with the service and visualize task reasoning results on the scene graph.

2. Or call the [ROS service](https://github.com/ntnu-arl/semantic_inference_ros/blob/master/semantic_inference_msgs/srv/NavigationPrompt.srv):  ```/semantic_inference/navigation_prompt_service/navigation_prompt```



---

## Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{puigjaner2026reasoninggraph,
    title={Relationship-Aware Hierarchical 3D Scene Graph},
    author={Gassol Puigjaner, Albert and Zacharia, Angelos and Alexis, Kostas},
    booktitle={2026 IEEE International Conference on Robotics and Automation (ICRA)}, 
    year={2026}
}
```

---

## License

Released under **BSD-3-Clause**.

---

## Acknowledgements

This open-source release is based on work supported by the **European Commission** through:

- **Project SYNERGISE**, under **Horizon Europe Grant Agreement No. 101121321**

---

## Contact

For questions or support, reach out via [GitHub Issues](https://github.com/ntnu-arl/reasoning_hydra/issues) or contact the authors directly:

- [Albert Gassol Puigjaner](mailto:albert.g.puigjaner@ntnu.no)
- [Angelos Zacharia](mailto:angelos.zacharia@ntnu.no)
- [Kostas Alexis](mailto:konstantinos.alexis@ntnu.no)
