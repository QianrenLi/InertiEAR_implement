# InertiEAR implementation

## Data Collector:

We implement an Android App to collect data, the app is called SpyApp.

The source code of app is https://github.com/EdisonE3/IMU-Collect-App

The code of this app we are put into spyapp.zip.

The data we used to train is placed in the `raw` directory.

Importing this project into Android Studio or IDEA, then you can compile it.

After that, you can install this app on an android mobile phone and use it collect data.

Open the app, click the right low icon into collect mode.

Then you need to push the button `test` to collect data.

The data format is shown as below:

```
103094561358952,0.32542386651039124,0.1782652884721756,10.971092224121094
103094561372962,0.26081767678260803,0.19980068504810333,10.643275260925293
103094562586243,0.14835500717163086,0.23808585107326508,10.059426307678223
...
```
First column is time stamp, the following three columns are data collected from x, y and z axises from the mobile phone.

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
