from pathlib import Path
import os
import shutil

data_path = Path(Path.cwd() / 'dslf4')
train_path = Path(Path.cwd() / 'train')
val_path = Path(Path.cwd() / 'val')
train_gt_path = Path(Path.cwd() / 'train' / 'gt')
train_input_path = Path(Path.cwd() / 'train' / 'input')
val_gt_path = Path(Path.cwd() / 'val' / 'gt')
val_input_path = Path(Path.cwd() / 'val' / 'input')

distance = 128
train_val_ratio = 0.7


def data_prepare(folder):
    """
     Process the data: loop through the folder (with names like '0001.png', '0002.png'...)
     Get the proper file names according to the distance provided. Move the files into
     input and gt folders.
    :param folder: the path to the folder you want to  process the data
    :return: does not return anything
    """
    # Make the folders
    if not train_path.exists():
        train_path.mkdir()
    if not val_path.exists():
        val_path.mkdir()
    if not train_gt_path.exists():
        train_gt_path.mkdir()
    if not val_gt_path.exists():
        val_gt_path.mkdir()
    if not train_input_path.exists():
        train_input_path.mkdir()
    if not val_input_path.exists():
        val_input_path.mkdir()

    file_names = [f for f in os.listdir(folder)]
    input_names = []
    gt_names = []

    counter = 0
    for filename in file_names:
        if counter == len(file_names) - distance:
            break
        name, extension = filename.split('.')
        first_frame = filename
        sec_name = int(name) + distance
        sec_frame = str(sec_name).zfill(4) + '.' + extension
        gt_frame = str((int(name) + sec_name) // 2).zfill(4) + '.' + extension
        input_names.append([first_frame, sec_frame])
        gt_names.append(gt_frame)
        counter += 1

    # for i, v in enumerate(input_names):
    #     print(v, gt_names[i])

    train_input = [f for f in input_names[:int(len(input_names)*train_val_ratio)]]
    train_gt = [f for f in gt_names[:int(len(gt_names)*train_val_ratio)]]
    val_input = [f for f in input_names[int(len(input_names)*train_val_ratio):]]
    val_gt = [f for f in gt_names[int(len(input_names)*train_val_ratio):]]
    print(len(train_input))
    print(len(val_input))

    # copy & rename the files for the train folder
    for idx, value in enumerate(sorted(train_input)):
        print(value, train_gt[idx])
        name, extension = value[0].split('.')
        first_frame_path = folder / value[0]
        sec_frame_path = folder / value[1]
        gt_frame_path = folder / train_gt[idx]

        new_first_frame_name = str(name) + '-first.' + extension
        new_sec_frame_name = str(name) + '-sec.' + extension
        new_first_frame_path = train_input_path / new_first_frame_name
        new_sec_frame_path = train_input_path / new_sec_frame_name
        new_gt_name = str(name) + '-gt.' + extension
        new_gt_path = train_gt_path / new_gt_name
        shutil.copyfile(first_frame_path, new_first_frame_path)
        shutil.copyfile(sec_frame_path, new_sec_frame_path)
        shutil.copyfile(gt_frame_path, new_gt_path)

    print('-------------------------------')

    # copy & rename the files for the val folder
    for idx, value in enumerate(val_input):
        print(value, val_gt[idx])
        name, extension = value[0].split('.')
        first_frame_path = folder / value[0]
        sec_frame_path = folder / value[1]
        gt_frame_path = folder / val_gt[idx]

        new_first_frame_name = str(name) + '-first.' + extension
        new_sec_frame_name = str(name) + '-sec.' + extension
        new_first_frame_path = val_input_path / new_first_frame_name
        new_sec_frame_path = val_input_path / new_sec_frame_name
        new_gt_name = str(name) + '-gt.' + extension
        new_gt_path = val_gt_path / new_gt_name
        shutil.copyfile(first_frame_path, new_first_frame_path)
        shutil.copyfile(sec_frame_path, new_sec_frame_path)
        shutil.copyfile(gt_frame_path, new_gt_path)

    print('-------------DONE-------------')


if __name__ == '__main__':
    data_prepare(data_path)

