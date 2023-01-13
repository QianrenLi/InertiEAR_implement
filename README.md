# InertiEAR implementation


## Model Training
+ After data collect, use `mian` function in `read_data.py`
```    in_dir_path = "files_train/original_data_new"
    out_dir_path = "files_train/signal_data_new"
    for file_name in os.listdir(in_dir_path):
        if file_name.count("acc"):
            acc_file = file_name
            gyr_file = file_name.replace("acc", "gyr")
            label = int(file_name.replace(".txt", "").split("_")[-2])
            print(acc_file, gyr_file, label)
            out_dir_path_i = out_dir_path + "/files_" + str(label)
            if not os.path.isdir(out_dir_path_i):
                os.mkdir(out_dir_path_i)
            data_processing(acc_path=in_dir_path + "/" + acc_file, gyr_path=in_dir_path + "/" + gyr_file,
                            file_directory=out_dir_path_i + "/", label=label)
```
where `in_dir_path` denotes the path where the original data exist and `out_dir_path` denotes the file in `.npy`
+ Run the `main` function in `Main.py` with modified training model can be defined personally like `myModel = SENet()`
```
    myModel = torch.load("model_good/new/mobile_net.pth")
    myModel = myModel.to(device)

    # Training Model
    num_epochs = 20
    training(myModel, train_dl, val_dl, num_epochs)

    # Inference
    correlation_matrix = inference(myModel, val_dl, is_correlation=True)
    print(correlation_matrix)

    torch.save(myModel,"model/mobile_net.pth")
```
Num of epoch recommended to less 20 epochs for SENet.py to avoid overffiting.
## Result Display
An implemented function `segmentation` in `read_data.py` can be used to display the segmentation process with input `is_plot = True`.
This function should called first by estiblish a segmentation handler as `h_seg = read_data.segmentation_handle(acc_xyz, gyr_xyz, acc_t, gyr_t should be , Fs = 400)`
where the `acc_xyz, gyr_xyz,acc_t,gyr_t` should be read by `read_data.signal_read(path)` where `path` is the cellphone sampled IMU data in `txt` format.
