### run foot_pose_box.py to get keypoint of the cropped foot box image
### run foot_pose_tfrecords.py to make tfrecords

https://arxiv.org/abs/2006.10204

Dependencies

    docker 20.10.17
    tensorflow 2.6.1
    tensorflowjs 3.9.0
    opencv 4.5.5.62


Installation

    docker script

        sudo docker run -it --rm  \
          --name "tf_$USER_$(date +%Y%m%d_%H%M%S)" \
          -v /etc/group:/etc/group:ro \
          -v /etc/passwd:/etc/passwd:ro \
          -v $HOME:$HOME \
          --privileged \
          --shm-size=256m \
          --net=host \
          -e DISPLAY=$DISPLAY \
          -env="QT_X11_NO_MITSHM=1" \
          -e XDG_RUNTIME_DIR=/run/user/1000/bin/bash \
          -w=$(pwd) \
          --gpus '"device=0"' \
          tensorflow/tensorflow:2.6.1-gpu \

          /bin/bash

    install dependencies

        pip install imgaug
        pip install numpy==1.23.5
        pip install silence_tensorflow
        pip install tensorflow_addons
        pip install tflite_support
        pip install tensorflowjs==3.9.0
        pip install opencv-python==4.5.5.62
        pip install pillow
        apt-get update
        apt-get -y install libgl1-mesa-glx



Execution

    training

        shell command
        chmod +x train.sh
        ./train.sh

    training strategy

        heatmap
            config file : configs/blazepose_cfg.py
            learning : 0.0004(~100 epoch) -> 0.00004(~150 epoch)
            howto : to change the learning, In config file(blazepose_cfg.py),
                    modify 'BASE_CFG['train']['load_weight_path']' to 100 epoch checkpoint file path,
                    modify BASE_CFG['train']['learning_rate']['init_learning_rate'] to next learning rate

        regression
            config file : configs/reg_cfg.py
            learning : 0.0001(~50epoch) -> 0.00001(~80epoch) -> 0.000001(~100epoch)
            howto : to start learning with regression mode, In 'train.sh' file,
                    change blazepose -> blazepose_reg,
                    and In config file(reg_cfg.py)
                    modify 'BASE_CFG['train']['load_weight_path']' to heatmap trained checkpoint file path

                    to change the learning, In config file(reg_cfg.py),
                    modify 'BASE_CFG['train']['load_weight_path']' to  epoch checkpoint file path,
                    modify BASE_CFG['train']['learning_rate']['init_learning_rate'] to next learning rate


    convert tensorflowjs

        need to change the paths(--input_ckpt, --save_folder, )

        sudo chmod +x ./model_export/tfjs_converter.sh
        ./model_export/tfjs_converter.sh

